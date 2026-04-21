"""Benchmark LLM-based manga translation methods on the OpenMantra dataset.

Methods:
  - baseline: return source text unchanged
  - llm_simple: translate each bubble via text-only LLM call (chapter-level batched)
  - llm_image: same but attaches the page image as visual context
  - llm_rolling: text-only translation with a running plot summary across pages
  - llm_image_rolling: attaches the page image AND a running plot summary

Resumes from a checkpoint file so crashes don't lose progress.
Computes Google BLEU (nltk gleu) plus sacrebleu BLEU and chrF.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from nltk.translate.gleu_score import corpus_gleu, sentence_gleu
from nltk.translate.meteor_score import meteor_score
import sacrebleu
from rouge_score import rouge_scorer
from tqdm import tqdm


ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "open-mantra-dataset"
ANNOT_PATH = DATASET_DIR / "annotation.json"

METHODS = (
    "baseline",
    "koharu",
    "koharu_rolling",
    "koharu_image",
    "koharu_image_rolling",
)

# ---------------------------- Prompts ------------------------------------

ROLLING_CONTEXT_PREFIX = (
    "Running plot summary so far (for context only, do not translate it):\n"
    "{summary}\n\n"
)

KOHARU_BLOCK_TAG_INSTRUCTIONS = (
    "The input uses numbered tags like [1], [2], etc. to mark each text block. "
    "Translate only the text after each tag. Keep every tag exactly unchanged, "
    "including numbers and order. Output the same tags followed by the "
    "translated text. Do not merge, split, or reorder blocks."
)

KOHARU_SYSTEM_PROMPT = (
    "You are a professional manga translator. Translate manga dialogue into "
    "natural {target_lang} that fits inside speech bubbles. Preserve character "
    "voice, emotional tone, relationship nuance, emphasis, and sound effects "
    "naturally. Keep the wording concise. Do not add notes, explanations, or "
    "romanization. " + KOHARU_BLOCK_TAG_INSTRUCTIONS
)

SUMMARY_PROMPT = (
    "You are maintaining a concise running plot summary of a manga chapter. "
    "Given the previous summary and the dialogue of the new page, produce an "
    "updated summary in at most 200 words in English. Output ONLY the summary text."
)

IMAGE_DESCRIPTION_PROMPT = (
    "Describe this manga page for a translator who cannot see the image, just the text. "
    "Cover: characters present (appearance, expressions, body language), "
    "setting and action, panel-to-panel flow in reading order, who is "
    "speaking in each panel and their emotional tone, and any visual "
    "context (sound effects, signs, objects) that affects dialogue meaning. "
    "Be concise but specific. Output only the description."
)

IMAGE_DESCRIPTION_CONTEXT_PREFIX = (
    "Visual description of the page (for context only, do not translate it):\n"
    "{description}\n\n"
)


# ---------------------------- Data ---------------------------------------

@dataclass
class Bubble:
    ja: str
    en: str
    zh: str


@dataclass
class Page:
    book: str
    page_index: int
    image_path: Path
    bubbles: list[Bubble]

    @property
    def key(self) -> str:
        return f"{self.book}/{self.page_index}"


def load_pages() -> list[Page]:
    data = json.loads(ANNOT_PATH.read_text())
    pages: list[Page] = []
    for book in data:
        for p in book["pages"]:
            bubbles = [
                Bubble(ja=t["text_ja"], en=t["text_en"], zh=t["text_zh"])
                for t in p["text"]
            ]
            img_rel = p["image_paths"]["ja"]
            pages.append(
                Page(
                    book=book["book_title"],
                    page_index=p["page_index"],
                    image_path=DATASET_DIR / img_rel,
                    bubbles=bubbles,
                )
            )
    return pages


# ---------------------------- Checkpoint --------------------------------

class Checkpoint:
    """JSONL-per-page checkpoint. Each line: {method, book, page_index, translations}."""

    def __init__(self, path: Path):
        self.path = path
        self.done: dict[tuple[str, str, int], list[str]] = {}
        self.models: dict[str, list[str]] = defaultdict(list)
        self.usage: dict[str, dict] = defaultdict(lambda: {"input": 0, "output": 0, "reasoning": 0})
        if path.exists():
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self.done[(rec["method"], rec["book"], rec["page_index"])] = rec["translations"]
                    if rec.get("model"):
                        self.models[rec["method"]].append(rec["model"])
                    if rec.get("usage"):
                        _add_usage(self.usage[rec["method"]], rec["usage"])

    def has(self, method: str, page: Page) -> bool:
        return (method, page.book, page.page_index) in self.done

    def get(self, method: str, page: Page) -> list[str]:
        return self.done[(method, page.book, page.page_index)]

    def put(self, method: str, page: Page, translations: list[str], extra: dict | None = None):
        rec = {
            "method": method,
            "book": page.book,
            "page_index": page.page_index,
            "translations": translations,
        }
        if extra:
            rec.update(extra)
            if extra.get("model"):
                self.models[method].append(extra["model"])
            if extra.get("usage"):
                _add_usage(self.usage[method], extra["usage"])
        self.done[(method, page.book, page.page_index)] = translations
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


# ---------------------------- LLM helpers --------------------------------

def encode_image(path: Path) -> str:
    b = path.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction from a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
        # remove trailing code fence leftovers
        if text.endswith("```"):
            text = text[:-3]
    # find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def chat(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int = 7000,
    temperature: float = 0.0,
    retries: int = 3,
) -> tuple[str, str, dict]:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort="medium",
            )
            usage = {"input": 0, "output": 0, "reasoning": 0}
            u = getattr(resp, "usage", None)
            if u is not None:
                usage["input"] = getattr(u, "prompt_tokens", 0) or 0
                usage["output"] = getattr(u, "completion_tokens", 0) or 0
                details = getattr(u, "completion_tokens_details", None)
                if details is not None:
                    reasoning = getattr(details, "reasoning_tokens", None)
                    if reasoning is None and isinstance(details, dict):
                        reasoning = details.get("reasoning_tokens", 0)
                    usage["reasoning"] = reasoning or 0
            return (resp.choices[0].message.content or "", resp.model or model, usage)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")


def _add_usage(acc: dict, add: dict) -> None:
    for k in ("input", "output", "reasoning"):
        acc[k] = acc.get(k, 0) + add.get(k, 0)


# ---------------------------- Methods ------------------------------------

def translate_baseline(page: Page, target: str) -> list[str]:
    return [b.ja for b in page.bubbles]

_KOHARU_TAG_RE = __import__("re").compile(r"^\s*\[(\d+)\]\s*(.*)$")

def _translate_koharu(
    client: OpenAI,
    model: str,
    page: Page,
    target_lang: str,
    attach_image: bool = False,
    extra_system: str = "",
) -> tuple[list[str], str | None, dict]:
    usage = {"input": 0, "output": 0, "reasoning": 0}
    if not page.bubbles:
        return [], None, usage
    tagged_input = "\n".join(f"[{i + 1}] {b.ja}" for i, b in enumerate(page.bubbles))
    system = KOHARU_SYSTEM_PROMPT.format(target_lang=target_lang)
    if extra_system:
        system = system + "\n\n" + extra_system
    if attach_image:
        user_content: Any = [
            {"type": "text", "text": tagged_input},
            {"type": "image_url", "image_url": {"url": encode_image(page.image_path)}},
        ]
    else:
        user_content = tagged_input
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    raw, resp_model, u = chat(client, model, messages)
    _add_usage(usage, u)

    out: dict[int, str] = {}
    for line in raw.splitlines():
        m = _KOHARU_TAG_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        text = m.group(2).strip()
        if 1 <= idx <= len(page.bubbles) and idx not in out:
            out[idx] = text
    return [out.get(i + 1, "") for i in range(len(page.bubbles))], resp_model, usage


def _update_summary(
    client: OpenAI,
    model: str,
    prev_summary: str,
    page: Page,
    translations: list[str],
    image_description: str = "",
) -> tuple[str, dict]:
    dialogue = "\n".join(
        f"- {ja}  =>  {en}" for ja, en in zip((b.ja for b in page.bubbles), translations)
    ) or "(no dialogue on this page)"
    user_text = f"Previous summary:\n{prev_summary or '(none)'}\n\n"
    if image_description:
        user_text += f"Visual description of the new page:\n{image_description}\n\n"
    user_text += f"New page dialogue:\n{dialogue}"
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": user_text},
    ]
    text, _, usage = chat(client, model, messages, max_tokens=7000, temperature=0.0)
    return text.strip(), usage


def _describe_image(
    client: OpenAI, model: str, page: Page
) -> tuple[str, dict]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": encode_image(page.image_path)}},
            ],
        },
    ]
    text, _, usage = chat(client, model, messages, max_tokens=3000, temperature=0.0)
    return text.strip(), usage


# ---------------------------- Runner -------------------------------------

def group_by_book(pages: list[Page]) -> dict[str, list[Page]]:
    g: dict[str, list[Page]] = defaultdict(list)
    for p in pages:
        g[p.book].append(p)
    for b in g:
        g[b].sort(key=lambda p: p.page_index)
    return g


def run_method(
    method: str,
    pages: list[Page],
    ckpt: Checkpoint,
    client: OpenAI | None,
    model: str,
    target_lang: str,
    target_key: str,
):
    by_book = group_by_book(pages)
    for book, bpages in by_book.items():
        summary = ""
        # If resuming rolling, we need to rebuild summary from stored extras; fall back
        # to re-summarizing from prior translations we already have.
        if method in ("llm_rolling", "llm_image_rolling", "koharu_rolling", "koharu_image_rolling", "koharu_image_description_rolling"):
            for p in bpages:
                if not ckpt.has(method, p):
                    break
                prev_trans = ckpt.get(method, p)
                # Rebuild summary (cheap, only needed until first non-done page)
                if client is not None:
                    summary, _ = _update_summary(client, model, summary, p, prev_trans)

        pbar = tqdm(bpages, desc=f"{method}:{book}", leave=False)
        for page in pbar:
            if ckpt.has(method, page):
                continue
            resp_model: str | None = None
            usage = {"input": 0, "output": 0, "reasoning": 0}
            if method == "baseline":
                out = translate_baseline(page, target_lang)
            elif method == "koharu":
                out, resp_model, usage = _translate_koharu(client, model, page, target_lang)
            elif method == "koharu_image":
                out, resp_model, usage = _translate_koharu(
                    client, model, page, target_lang, attach_image=True
                )
            elif method in ("koharu_rolling", "koharu_image_rolling"):
                extra = ROLLING_CONTEXT_PREFIX.format(summary=summary) if summary else ""
                out, resp_model, usage = _translate_koharu(
                    client, model, page, target_lang,
                    attach_image=(method == "koharu_image_rolling"),
                    extra_system=extra,
                )
                summary, sum_usage = _update_summary(client, model, summary, page, out)
                _add_usage(usage, sum_usage)
            elif method == "koharu_image_description":
                description, desc_usage = _describe_image(client, model, page)
                _add_usage(usage, desc_usage)
                extra = IMAGE_DESCRIPTION_CONTEXT_PREFIX.format(description=description)
                out, resp_model, tr_usage = _translate_koharu(
                    client, model, page, target_lang,
                    attach_image=False,
                    extra_system=extra,
                )
                _add_usage(usage, tr_usage)
            elif method == "koharu_image_description_rolling":
                description, desc_usage = _describe_image(client, model, page)
                _add_usage(usage, desc_usage)
                extra = IMAGE_DESCRIPTION_CONTEXT_PREFIX.format(description=description)
                if summary:
                    extra = extra + ROLLING_CONTEXT_PREFIX.format(summary=summary)
                out, resp_model, tr_usage = _translate_koharu(
                    client, model, page, target_lang,
                    attach_image=False,
                    extra_system=extra,
                )
                _add_usage(usage, tr_usage)
                summary, sum_usage = _update_summary(
                    client, model, summary, page, out, image_description=description
                )
                _add_usage(usage, sum_usage)
            else:
                raise ValueError(method)
            extras: dict = {}
            if resp_model is not None:
                extras["model"] = resp_model
            if method in ("koharu_rolling", "koharu_image_rolling"):
                extras["summary"] = summary
            if method != "baseline":
                extras["usage"] = usage
            ckpt.put(method, page, out, extra=extras or None)


# ---------------------------- Metrics ------------------------------------

def tokenize(s: str) -> list[str]:
    # Simple whitespace + basic punct split; sacrebleu has its own tokenizer for BLEU
    return s.strip().split()


def score_method(method: str, pages: list[Page], ckpt: Checkpoint, target_key: str) -> dict:
    refs: list[str] = []
    hyps: list[str] = []
    per_page_gleu: list[float] = []
    missing = 0
    for p in pages:
        if not ckpt.has(method, p):
            missing += len(p.bubbles)
            continue
        outs = ckpt.get(method, p)
        for bubble, hyp in zip(p.bubbles, outs):
            ref = getattr(bubble, target_key)
            refs.append(ref)
            hyps.append(hyp)
    models = sorted(set(ckpt.models.get(method, [])))
    usage = dict(ckpt.usage.get(method, {"input": 0, "output": 0, "reasoning": 0}))
    if not refs:
        return {"n": 0, "missing_bubbles": missing, "models": models, "usage": usage}

    # Google BLEU (gleu) — corpus & per-sentence
    ref_tok = [[tokenize(r)] for r in refs]
    hyp_tok = [tokenize(h) for h in hyps]
    gleu_corpus = corpus_gleu(ref_tok, hyp_tok)
    gleu_sent = [sentence_gleu([tokenize(r)], tokenize(h)) for r, h in zip(refs, hyps)]

    # sacrebleu BLEU + chrF
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])

    # METEOR (sentence-level, averaged)
    meteor_scores = [
        meteor_score([ref_t], hyp_t) for ref_t, hyp_t in zip((t[0] for t in ref_tok), hyp_tok)
    ]
    meteor_avg = sum(meteor_scores) / len(meteor_scores)

    # ROUGE-L (sentence-level F1, averaged)
    rl_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = [rl_scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]
    rouge_l_avg = sum(rouge_l_scores) / len(rouge_l_scores)

    avg_ref_len = sum(len(tokenize(r)) for r in refs) / len(refs)
    avg_hyp_len = sum(len(tokenize(h)) for h in hyps) / len(hyps)

    return {
        "n": len(refs),
        "missing_bubbles": missing,
        "models": models,
        "gleu_corpus": round(gleu_corpus, 4),
        "gleu_sentence_avg": round(sum(gleu_sent) / len(gleu_sent), 4),
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "meteor": round(meteor_avg, 4),
        "rouge_l": round(rouge_l_avg, 4),
        "avg_ref_tokens": round(avg_ref_len, 2),
        "avg_hyp_tokens": round(avg_hyp_len, 2),
        "usage": usage,
    }


# ---------------------------- Main ---------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://192.168.1.131:5001/v1/")) # https://api.openai.com/v1
    ap.add_argument("--api-key", default="") # os.environ.get("OPENAI_API_KEY", "sk-none")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--target-lang", default="English", help="Target language name for prompts")
    ap.add_argument(
        "--target-key",
        default="en",
        choices=["en", "zh"],
        help="Reference field to score against",
    )
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--books", nargs="*", default=None, help="Subset of book titles to run")
    ap.add_argument("--max-pages", type=int, default=None, help="Limit pages per book (debug)")
    ap.add_argument("--checkpoint", default="results/checkpoint.jsonl")
    ap.add_argument("--report", default="results/report.json")
    ap.add_argument("--score-only", action="store_true", help="Skip translation, just score checkpoint")
    args = ap.parse_args()

    pages = load_pages()
    if args.books:
        pages = [p for p in pages if p.book in set(args.books)]
    if args.max_pages:
        by_book = group_by_book(pages)
        pages = [p for b in by_book.values() for p in b[: args.max_pages]]

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(ckpt_path)

    client: OpenAI | None = None
    if not args.score_only and any(m != "baseline" for m in args.methods):
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        print(f"Using base_url={args.base_url} model={args.model}")

    if not args.score_only:
        for method in args.methods:
            print(f"\n=== Running {method} ===")
            run_method(method, pages, ckpt, client, args.model, args.target_lang, args.target_key)

    # Score
    print("\n=== Scores ===")
    report = {"config": vars(args), "scores": {}}
    for method in args.methods:
        s = score_method(method, pages, ckpt, args.target_key)
        report["scores"][method] = s
        print(f"{method:>12}: {s}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
