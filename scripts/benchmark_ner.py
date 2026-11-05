"""Zero-shot BIO tagging of cement abstracts via a vLLM server.

Each request is one sentence (tokens only, no gold labels and no worked
example). Temperature is 0. The model is told to emit token/tag lines and
nothing else. Scores come from MatSciBERT's conlleval.py.

Example:
  python scripts/benchmark_ner.py --model qwen25-0.5b --base-url http://localhost:8000/v1
  python scripts/benchmark_ner.py --report-only
"""

import argparse
import csv
import json
import os
import re
import sys
import time

from benchmark_mineralogy_cls import MODELS, env_prefix, repo_root


def _conlleval_dir():
    candidates = [
        os.environ.get("CONLLEVAL_DIR"),
        os.path.join(repo_root(), "third_party", "conlleval"),
        os.path.join(repo_root(), "scripts"),
    ]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "conlleval.py")):
            return path
    raise SystemExit(
        "conlleval.py not found. Place it under third_party/conlleval or scripts, "
        "or set CONLLEVAL_DIR to its directory."
    )


sys.path.insert(0, _conlleval_dir())
import conlleval  # noqa: E402

NER_MODELS = (
    "qwen25-0.5b",
    "gemma3-270m",
    "ministral-3b",
    "qwen25-7b",
    "gemma3-4b",
    "ministral-8b",
    "llamat-2-chat",
    "llamat-3-chat",
)

SYSTEM_PROMPT = "You are construction materials expert."

TAGS = (
    "O",
    "B-MAT",
    "I-MAT",
    "B-PRO",
    "I-PRO",
    "B-CMT",
    "I-CMT",
    "B-SMT",
    "I-SMT",
    "B-DSC",
    "I-DSC",
    "B-APL",
    "I-APL",
    "B-SPL",
    "I-SPL",
)
TAG_SET = set(TAGS)
TAG_RE = re.compile(r"\b(?:O|[BI]-(?:MAT|PRO|CMT|SMT|DSC|APL|SPL))\b")
ENTITY_TYPES = ("MAT", "PRO", "CMT", "SMT", "DSC", "APL", "SPL")

USER_PROMPT = """Label every token in the sentence with exactly one tag from this closed set:
{tags}

B- marks the first token of an entity. I- marks a continuation of the same entity. O marks a token outside any entity.

Return only one line per token, in the same order, formatted as:
token TAG

Do not explain. Do not think aloud. Do not add any other text.

Sentence:
{sentence}"""

LLAMAT_USER_PROMPT = """Label every token in the sentence with exactly one tag from this closed set:
{tags}

B- marks the first token of an entity. I- marks a continuation of the same entity. O marks a token outside any entity.

Return only the tags, one per line, in the same order as the tokens. Do not repeat the tokens. Do not explain.

Sentence:
{sentence}"""


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot NER benchmark")
    parser.add_argument("--model", choices=NER_MODELS, default=None)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(repo_root(), "data", "ner"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(repo_root(), "results", "ner"),
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def load_corpus(data_dir):
    documents = []
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(data_dir, name)
        with open(path, encoding="utf-8") as handle:
            text = handle.read().strip()
        sentences = []
        for block in re.split(r"\n\s*\n", text):
            tokens, labels = [], []
            for line in block.split("\n"):
                parts = re.split(r" +", line)
                if len(parts) != 2:
                    tokens, labels = [], []
                    break
                token, label = parts
                if not token:
                    token = " "
                if not label:
                    label = "O"
                tokens.append(token)
                labels.append(label)
            if tokens:
                sentences.append({"tokens": tokens, "labels": labels})
        documents.append({"file": name, "sentences": sentences})
    return documents


def extract_tags(text, count):
    line_tags = []
    for line in (text or "").splitlines():
        parts = line.split()
        if parts and parts[-1] in TAG_SET:
            line_tags.append(parts[-1])
    found = [tag for tag in TAG_RE.findall(text or "") if tag in TAG_SET]
    tags = line_tags if len(line_tags) >= len(found) else found
    if len(tags) >= count:
        return tags[:count]
    return tags + ["O"] * (count - len(tags))


def output_token_limit(model, n_tokens):
    # One tag per token. A tag such as B-MAT can be two pieces, so LLaMat gets
    # two output tokens per input token and then stops.
    if model.startswith("llamat"):
        return max(2 * n_tokens, 2)
    return min(4096, max(64, 6 * n_tokens + 32))


def llamat_prompt(model, question):
    # Same wrappers as evaluation_codes/mof2_run.py in M3RG-IITD/llamat.
    if model == "llamat-2-chat":
        return (
            " [INST]<<SYS>>\n"
            + SYSTEM_PROMPT
            + "<</SYS>>\n"
            + question
            + "\n[/INST]"
        )
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        + SYSTEM_PROMPT
        + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        + question
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def request_tags(session, args, model_id, sentence, extra_body):
    import requests

    n_tokens = len(sentence.split())
    question = (
        LLAMAT_USER_PROMPT if args.model.startswith("llamat") else USER_PROMPT
    ).format(tags=" ".join(TAGS), sentence=sentence)
    if args.model.startswith("llamat"):
        payload = {
            "model": model_id,
            "prompt": llamat_prompt(args.model, question),
            "temperature": 0.0,
            "max_tokens": output_token_limit(args.model, n_tokens),
            "add_special_tokens": False,
        }
        url = f"{args.base_url.rstrip('/')}/completions"
        content_key = "text"
    else:
        payload = {
            "model": model_id,
            "temperature": 0.0,
            "max_tokens": output_token_limit(args.model, n_tokens),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        }
        if extra_body:
            payload.update(extra_body)
        url = f"{args.base_url.rstrip('/')}/chat/completions"
        content_key = "message"
    headers = {"Content-Type": "application/json"}
    last_error = None
    for attempt in range(args.max_retries):
        try:
            started = time.time()
            response = session.post(url, headers=headers, json=payload, timeout=args.timeout)
            if response.status_code == 400 and "reasoning_effort" in payload:
                payload.pop("reasoning_effort", None)
                response = session.post(url, headers=headers, json=payload, timeout=args.timeout)
            response.raise_for_status()
            latency = time.time() - started
            body = response.json()
            choice = body["choices"][0]
            if content_key == "text":
                content = choice.get("text") or ""
            else:
                content = (choice.get("message") or {}).get("content") or ""
            return content, latency, None
        except Exception as exc:  # noqa: BLE001 - retry transport and server errors
            last_error = exc
            if attempt < args.max_retries - 1:
                time.sleep(min(2**attempt, 15))
    return "", 0.0, last_error


def read_completed(path):
    completed = {}
    if not os.path.exists(path):
        return completed
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (row.get("file"), row.get("sentence"))
            if key[0] and isinstance(key[1], int):
                completed[key] = row
    return completed


def conll_lines(records):
    lines = []
    for record in records:
        for token, gold, pred in zip(record["tokens"], record["gold"], record["pred"]):
            lines.append(f"{token} {gold} {pred}")
        lines.append("-X- O O")
    return lines


def score_records(records):
    counts = conlleval.evaluate(conll_lines(records))
    overall, by_type = conlleval.metrics(counts)
    return counts, overall, by_type


def write_conll(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(conll_lines(records)))
        handle.write("\n")


def load_records(path):
    rows = list(read_completed(path).values())
    rows.sort(key=lambda row: (row.get("file", ""), row.get("sentence", 0)))
    return rows


def write_report(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summaries = []
    md = ["# Cement NER", ""]
    for key in NER_MODELS:
        records = load_records(os.path.join(output_dir, key, "predictions.jsonl"))
        if not records:
            continue
        counts, overall, by_type = score_records(records)
        write_conll(os.path.join(output_dir, key, "predictions.conll"), records)
        row = {
            "model": key,
            "model_id": MODELS[key]["model_id"],
            "precision": overall.prec,
            "recall": overall.rec,
            "f1": overall.fscore,
            "tokens": counts.token_counter,
            "phrases": counts.found_correct,
        }
        for entity in ENTITY_TYPES:
            metrics = by_type.get(entity)
            row[f"{entity}_precision"] = metrics.prec if metrics else 0.0
            row[f"{entity}_recall"] = metrics.rec if metrics else 0.0
            row[f"{entity}_f1"] = metrics.fscore if metrics else 0.0
        summaries.append(row)
        md.append(f"## {key}")
        md.append("")
        md.append(conlleval.report(counts).rstrip())
        md.append("")

    fields = ["model", "model_id", "precision", "recall", "f1", "tokens", "phrases"]
    for entity in ENTITY_TYPES:
        fields.extend([f"{entity}_precision", f"{entity}_recall", f"{entity}_f1"])
    csv_path = os.path.join(output_dir, "report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            out = dict(row)
            for name, value in row.items():
                if isinstance(value, float):
                    out[name] = f"{value:.4f}"
            writer.writerow(out)
    md_path = os.path.join(output_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(md) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def run_model(args):
    import requests

    spec = MODELS[args.model]
    documents = load_corpus(args.data_dir)
    sentence_count = sum(len(doc["sentences"]) for doc in documents)
    model_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)
    pred_path = os.path.join(model_dir, "predictions.jsonl")
    completed = {} if args.overwrite else read_completed(pred_path)
    print(f"{args.model}: {sentence_count} sentences, {len(completed)} done")

    extra_body = {}
    if args.model == "ministral-3b":
        extra_body = {"reasoning_effort": "none"}

    session = requests.Session()
    mode = "w" if args.overwrite else "a"
    with open(pred_path, mode, encoding="utf-8") as handle:
        for document in documents:
            for index, sentence in enumerate(document["sentences"]):
                key = (document["file"], index)
                if key in completed:
                    continue
                text = " ".join(sentence["tokens"])
                raw, latency, error = request_tags(
                    session, args, spec["model_id"], text, extra_body
                )
                pred = extract_tags("" if error else raw, len(sentence["tokens"]))
                record = {
                    "file": document["file"],
                    "sentence": index,
                    "tokens": sentence["tokens"],
                    "gold": sentence["labels"],
                    "pred": pred,
                    "raw": raw,
                    "latency": round(latency, 3),
                    "model": args.model,
                    "model_id": spec["model_id"],
                    "error": None if error is None else f"{type(error).__name__}: {error}",
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(
                    f"  {document['file']}#{index} "
                    f"n={len(sentence['tokens'])} ({latency:.1f}s)"
                )

    records = load_records(pred_path)
    _counts, overall, _by_type = score_records(records)
    print(
        f"{args.model}: precision={overall.prec:.3f} "
        f"recall={overall.rec:.3f} f1={overall.fscore:.3f}"
    )
    write_report(args.output_dir)


def main():
    args = parse_args()
    if args.report_only:
        write_report(args.output_dir)
        return
    if not args.model:
        raise SystemExit("--model is required unless --report-only is set")
    run_model(args)


if __name__ == "__main__":
    main()
