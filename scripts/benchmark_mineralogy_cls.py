"""Zero-shot mineralogical-composition classification via a vLLM server.

The server is started by scripts/run_mineralogy_benchmark.sh. This client
sends each row of data/cls/test.csv and scores precision, recall, and F1
for the positive class (label 1).

Example:
  python scripts/benchmark_mineralogy_cls.py \
    --model qwen25-0.5b \
    --base-url http://localhost:8000/v1
  python scripts/benchmark_mineralogy_cls.py --report-only
"""

import argparse
import ast
import csv
import json
import os
import re
import shlex
import time

SYSTEM_PROMPT = "You are a cement chemistry expert."

USER_PROMPT = """The following text is a table caption followed by a table.

Return 1 if the text reports the mineralogical composition of cement, clinker, or related materials. Return 0 otherwise.

Reply with a single character: 0 or 1.

Text:
{text}"""

# Llamat repeats once generation is longer than the label. Each variant is
# capped at the number of tokens in one task answer (YES/NO, 0/1, or one JSON
# object). The sweep keeps the fully parsed variant with the highest F1.
LLAMAT_PROMPTS = {
    "yesno": {
        "max_tokens": 1,
        "style": "yesno_exact",
        "stop": ["<|im_end|>", "\n"],
        "template": """Does the text report the mineralogical composition of cement, clinker, or a related material?
Answer with YES or NO and nothing else.
Text:
{text}
Answer:""",
    },
    "dict": {
        "max_tokens": 12,
        "style": "json_exact",
        "stop": ["<|im_end|>"],
        "template": """Does the text report the mineralogical composition of cement, clinker, or a related material?
Reply with only {{'answer': 'YES'}} or {{'answer': 'NO'}} and then stop.
Do not list phase fractions.
Text:
{text}
""",
    },
    "json": {
        "max_tokens": 12,
        "style": "json_exact",
        "stop": ["<|im_end|>"],
        "template": """Does the text report the mineralogical composition of cement, clinker, or a related material?
Reply with only {{"answer":"YES"}} or {{"answer":"NO"}} and then stop.
Do not list phase fractions.
Text:
{text}
""",
    },
}
LLAMAT_PROMPT_ORDER = ("yesno", "dict", "json")
ANSWER_RE = re.compile(
    r"""^\s*\{?\s*['\"]?answer['\"]?\s*[:=]\s*\[?\s*['\"]?(yes|no)\b""",
    re.IGNORECASE,
)

MODELS = {
    "qwen25-0.5b": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "conda_env": "qwen_vllm",
        "env_root": "CONDA_ENV_ROOT",
    },
    "gemma3-270m": {
        "model_id": "google/gemma-3-270m-it",
        "conda_env": "gemma_cu128",
        "env_root": "SCRATCH_ENV_ROOT",
    },
    "ministral-3b": {
        "model_id": "mistralai/Ministral-3-3B-Instruct-2512",
        "conda_env": "mistral_cu128",
        "env_root": "SCRATCH_ENV_ROOT",
        # Native Mistral checkpoint (tekken + consolidated.safetensors).
        "extra_serve_args": [
            "--tokenizer-mode",
            "mistral",
            "--config-format",
            "mistral",
            "--load-format",
            "mistral",
        ],
    },
    "qwen25-7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "conda_env": "qwen_vllm",
        "env_root": "CONDA_ENV_ROOT",
    },
    "gemma3-4b": {
        "model_id": "google/gemma-3-4b-it",
        "conda_env": "gemma_cu128",
        "env_root": "SCRATCH_ENV_ROOT",
    },
    "ministral-8b": {
        "model_id": "mistralai/Ministral-8B-Instruct-2410",
        "conda_env": "mistral_cu128",
        "env_root": "SCRATCH_ENV_ROOT",
        "extra_serve_args": ["--tokenizer-mode", "mistral"],
    },
    # Weights are stored in float32. Cast to bfloat16 so an 8B model fits
    # on one A100 alongside the 16384-token context used for the tables.
    "llamat-2-chat": {
        "model_id": "m3rg-iitd/llamat-2-chat",
        "conda_env": "qwen_vllm",
        "env_root": "CONDA_ENV_ROOT",
        "extra_serve_args": ["--dtype", "bfloat16"],
        "max_model_len": 2048,
        "prompt_variants": True,
        "overwrite": True,
    },
    "llamat-3-chat": {
        "model_id": "m3rg-iitd/llamat-3-chat",
        "conda_env": "qwen_vllm",
        "env_root": "CONDA_ENV_ROOT",
        "extra_serve_args": ["--dtype", "bfloat16"],
        "max_model_len": 2048,
        "prompt_variants": True,
        "overwrite": True,
    },
}

LABEL_RE = re.compile(r"[01]")
YES_NO_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def env_prefix(spec):
    conda_root = os.environ.get("CONDA_ENV_ROOT") or os.environ.get("SCRATCH_ENV_ROOT")
    if not conda_root:
        raise SystemExit(
            "Set CONDA_ENV_ROOT or SCRATCH_ENV_ROOT to the directory that holds the vLLM environments."
        )
    scratch_root = os.environ.get("SCRATCH_ENV_ROOT", conda_root)
    root = conda_root if spec["env_root"] == "CONDA_ENV_ROOT" else scratch_root
    return os.path.join(root, spec["conda_env"])


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark mineralogy table classification")
    parser.add_argument("--model", choices=sorted(MODELS), default=None)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument(
        "--csv",
        default=os.path.join(repo_root(), "data", "cls", "test.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(repo_root(), "results", "mineralogy_cls"),
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild report.csv and report.md from existing predictions",
    )
    parser.add_argument(
        "--emit",
        choices=["shell"],
        default=None,
        help="Print serving settings for --model and exit",
    )
    return parser.parse_args()


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, row in enumerate(reader):
            rows.append(
                {
                    "index": index,
                    "text": row["text"],
                    "label": int(row["Label"]),
                }
            )
    return rows


def _yes_no_value(value):
    text = str(value).strip().upper()
    if text in {"YES", "1", "TRUE"}:
        return 1
    if text in {"NO", "0", "FALSE"}:
        return 0
    return None


def parse_label(text, style="digit"):
    if text is None:
        return None
    text = str(text).strip()
    if style == "yesno_exact":
        line = text.strip().splitlines()[0] if text.strip() else ""
        # A 2-token cap keeps the label and the first piece of <|im_end|>.
        line = re.sub(r"<\|?.*$", "", line)
        line = line.strip().strip(" .\"'<")
        if line.upper() == "YES":
            return 1
        if line.upper() == "NO":
            return 0
        return None
    if style == "digit_exact":
        text = text.strip()
        if text in {"0", "1"}:
            return int(text)
        return None
    if style == "json_exact":
        match = ANSWER_RE.search(text)
        if match is None:
            return None
        return 1 if match.group(1).upper() == "YES" else 0
    if style == "yesno":
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            blob = text[start : end + 1]
            obj = None
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                try:
                    obj = ast.literal_eval(blob)
                except (SyntaxError, ValueError):
                    obj = None
            if isinstance(obj, dict):
                for key in ("answer", "label"):
                    if key in obj:
                        parsed = _yes_no_value(obj[key])
                        if parsed is not None:
                            return parsed
        match = YES_NO_RE.search(text)
        if match is None:
            return None
        return 1 if match.group(1).upper() == "YES" else 0
    match = LABEL_RE.search(text)
    if match is None:
        return None
    return int(match.group(0))


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
            if isinstance(row.get("index"), int):
                completed[row["index"]] = row
    return completed


def llamat_prompt(question):
    # Official LLaMat eval wrapper: system / question / answer, not user / assistant.
    # https://github.com/M3RG-IITD/llamat/blob/main/evaluation_codes/mof2_run.py
    return (
        "<|im_start|>system\n"
        + SYSTEM_PROMPT
        + "<|im_end|>\n"
        "<|im_start|>question\n"
        + question
        + "<|im_end|>\n"
        "<|im_start|>answer\n"
    )


def request_label(session, args, spec, text, variant=None):
    user_prompt = variant["template"] if variant else spec.get("user_prompt", USER_PROMPT)
    max_tokens = variant["max_tokens"] if variant else spec.get("max_tokens", args.max_tokens)
    if variant is not None:
        payload = {
            "model": spec["model_id"],
            "prompt": llamat_prompt(user_prompt.format(text=text)),
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stop": variant.get("stop", ["<|im_end|>"]),
        }
        url = f"{args.base_url.rstrip('/')}/completions"
    else:
        payload = {
            "model": spec["model_id"],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt.format(text=text)},
            ],
        }
        url = f"{args.base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    last_error = None
    for attempt in range(args.max_retries):
        try:
            started = time.time()
            response = session.post(url, headers=headers, json=payload, timeout=args.timeout)
            response.raise_for_status()
            latency = time.time() - started
            body = response.json()
            choice = body["choices"][0]
            if variant is not None:
                content = choice.get("text") or ""
            else:
                content = choice["message"].get("content") or ""
            return content, latency, None
        except Exception as exc:  # noqa: BLE001 - retry transport and server errors
            last_error = exc
            if attempt < args.max_retries - 1:
                time.sleep(min(2**attempt, 15))
    return "", 0.0, last_error


def score_rows(rows):
    tp = fp = fn = 0
    unparsed = 0
    for row in rows:
        gold = int(row["label"])
        parsed = row.get("parsed")
        if parsed is None:
            unparsed += 1
            predicted = 0
        else:
            predicted = int(parsed)
        if predicted == 1 and gold == 1:
            tp += 1
        elif predicted == 1 and gold == 0:
            fp += 1
        elif predicted == 0 and gold == 1:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    support = sum(int(row["label"]) == 1 for row in rows)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support_positive": support,
        "n": len(rows),
        "n_unparsed": unparsed,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def load_predictions(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda row: row.get("index", 0))
    return rows


def write_report(output_dir):
    summaries = []
    for key in MODELS:
        path = os.path.join(output_dir, key, "predictions.jsonl")
        rows = load_predictions(path)
        if not rows:
            continue
        metrics = score_rows(rows)
        prompt = rows[0].get("prompt")
        metrics["model"] = f"{key}/{prompt}" if prompt else key
        metrics["model_id"] = MODELS[key]["model_id"]
        summaries.append(metrics)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "report.csv")
    md_path = os.path.join(output_dir, "report.md")
    fields = [
        "model",
        "model_id",
        "precision",
        "recall",
        "f1",
        "support_positive",
        "n",
        "n_unparsed",
        "tp",
        "fp",
        "fn",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            out = dict(row)
            for name in ("precision", "recall", "f1"):
                out[name] = f"{row[name]:.4f}"
            writer.writerow(out)

    lines = [
        "# Mineralogy classification",
        "",
        "| model | checkpoint | precision | recall | f1 | positive | n | unparsed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {model} | `{model_id}` | {precision:.3f} | {recall:.3f} | {f1:.3f} | {support_positive} | {n} | {n_unparsed} |".format(
                **row
            )
        )
    if not summaries:
        lines.append("| | | | | | | | |")
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return summaries


def emit_shell(model_key):
    spec = MODELS[model_key]
    prefix = env_prefix(spec)
    extra = list(spec.get("extra_serve_args") or [])
    lines = [
        f"MODEL_KEY={shlex.quote(model_key)}",
        f"MODEL_ID={shlex.quote(spec['model_id'])}",
        f"CONDA_ENV={shlex.quote(spec['conda_env'])}",
        f"ENV_PREFIX={shlex.quote(prefix)}",
        f"PYTHON_BIN={shlex.quote(os.path.join(prefix, 'bin', 'python'))}",
        f"VLLM_BIN={shlex.quote(os.path.join(prefix, 'bin', 'vllm'))}",
        f"MAX_MODEL_LEN={int(spec.get('max_model_len', 16384))}",
        f"MAX_TOKENS={int(spec.get('max_tokens', 16))}",
        f"OVERWRITE={1 if spec.get('overwrite') else 0}",
        "EXTRA_SERVE_ARGS=({})".format(" ".join(shlex.quote(arg) for arg in extra)),
    ]
    print("\n".join(lines))


def load_chat_tokenizer(model_id):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.chat_template and "llamat-2" in model_id:
        template_path = os.path.join(repo_root(), "scripts", "llamat2_chat.jinja")
        with open(template_path, encoding="utf-8") as handle:
            tokenizer.chat_template = handle.read()
    return tokenizer


def fit_table_text(tokenizer, template, text, limit):
    def n_tokens(body):
        prompt = llamat_prompt(template.format(text=body))
        return len(tokenizer.encode(prompt, add_special_tokens=False))

    if n_tokens(text) <= limit:
        return text, False
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        if n_tokens(text[:mid]) <= limit:
            best = text[:mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return best, True


def write_llamat_sweep_summary(output_dir):
    lines = [
        "# Llamat prompt variants",
        "",
        "Generation is capped at the length of one label. A variant is usable when every row parses.",
        "",
        "| model | prompt | max tokens | precision | recall | f1 | parsed | unparsed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, spec in MODELS.items():
        if not spec.get("prompt_variants"):
            continue
        for name in LLAMAT_PROMPT_ORDER:
            path = os.path.join(output_dir, key, "prompts", name, "predictions.jsonl")
            rows = load_predictions(path)
            if not rows:
                continue
            metrics = score_rows(rows)
            parsed = metrics["n"] - metrics["n_unparsed"]
            lines.append(
                "| {model} | {prompt} | {max_tokens} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {parsed} | {unparsed} |".format(
                    model=key,
                    prompt=name,
                    max_tokens=LLAMAT_PROMPTS[name]["max_tokens"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    f1=metrics["f1"],
                    parsed=parsed,
                    unparsed=metrics["n_unparsed"],
                )
            )
    lines.append("")
    path = os.path.join(output_dir, "llamat_prompt_sweep.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Wrote {path}")


def run_llamat_sweep(args, spec):
    import requests

    tokenizer = load_chat_tokenizer(spec["model_id"])
    rows = load_rows(args.csv)
    model_dir = os.path.join(args.output_dir, args.model)
    session = requests.Session()
    usable = []
    for name in LLAMAT_PROMPT_ORDER:
        variant = LLAMAT_PROMPTS[name]
        limit = int(spec["max_model_len"]) - int(variant["max_tokens"]) - 1
        variant_dir = os.path.join(model_dir, "prompts", name)
        os.makedirs(variant_dir, exist_ok=True)
        pred_path = os.path.join(variant_dir, "predictions.jsonl")
        print(
            f"{args.model}/{name}: {len(rows)} rows, max_tokens={variant['max_tokens']}"
        )
        with open(pred_path, "w", encoding="utf-8") as handle:
            for row in rows:
                fitted, truncated = fit_table_text(
                    tokenizer, variant["template"], row["text"], limit
                )
                raw, latency, error = request_label(
                    session, args, spec, fitted, variant=variant
                )
                parsed = None if error else parse_label(raw, variant["style"])
                record = {
                    "index": row["index"],
                    "label": row["label"],
                    "raw": raw,
                    "parsed": parsed,
                    "truncated": truncated,
                    "prompt": name,
                    "max_tokens": variant["max_tokens"],
                    "latency": round(latency, 3),
                    "model": args.model,
                    "model_id": spec["model_id"],
                    "error": None if error is None else f"{type(error).__name__}: {error}",
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                status = "ERR" if error else str(parsed)
                print(
                    f"  [{row['index']:02d}] gold={row['label']} pred={status} "
                    f"raw={raw!r} ({latency:.1f}s)"
                )
        scored = load_predictions(pred_path)
        metrics = score_rows(scored)
        print(
            f"{args.model}/{name}: precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
            f"unparsed={metrics['n_unparsed']}"
        )
        if metrics["n"] == len(rows) and metrics["n_unparsed"] == 0:
            usable.append((metrics["f1"], -LLAMAT_PROMPT_ORDER.index(name), name, scored))

    if usable:
        usable.sort(reverse=True)
        _, _, name, scored = usable[0]
        canonical = os.path.join(model_dir, "predictions.jsonl")
        with open(canonical, "w", encoding="utf-8") as handle:
            for record in scored:
                handle.write(json.dumps(record) + "\n")
        print(f"{args.model}: selected prompt {name} ({len(scored)} parsable rows)")
    else:
        print(f"{args.model}: no prompt parsed every row")
    write_llamat_sweep_summary(args.output_dir)
    write_report(args.output_dir)


def run_model(args):
    spec = MODELS[args.model]
    if spec.get("prompt_variants"):
        run_llamat_sweep(args, spec)
        return
    import requests
    rows = load_rows(args.csv)
    model_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)
    pred_path = os.path.join(model_dir, "predictions.jsonl")
    completed = {} if args.overwrite else read_completed(pred_path)
    pending = [row for row in rows if row["index"] not in completed]
    print(f"{args.model}: {len(rows)} rows, {len(pending)} pending")

    mode = "w" if args.overwrite else "a"
    session = requests.Session()
    with open(pred_path, mode, encoding="utf-8") as handle:
        for row in pending:
            raw, latency, error = request_label(session, args, spec, row["text"])
            parsed = None if error else parse_label(raw, spec.get("label_style", "digit"))
            record = {
                "index": row["index"],
                "label": row["label"],
                "raw": raw,
                "parsed": parsed,
                "latency": round(latency, 3),
                "model": args.model,
                "model_id": spec["model_id"],
                "error": None if error is None else f"{type(error).__name__}: {error}",
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            status = "ERR" if error else str(parsed)
            print(f"  [{row['index']:02d}] gold={row['label']} pred={status} ({latency:.1f}s)")

    scored = load_predictions(pred_path)
    metrics = score_rows(scored)
    print(
        f"{args.model}: precision={metrics['precision']:.3f} "
        f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
        f"unparsed={metrics['n_unparsed']}"
    )
    write_report(args.output_dir)


def main():
    args = parse_args()
    if args.emit:
        if not args.model:
            raise SystemExit("--emit requires --model")
        emit_shell(args.model)
        return
    if args.report_only:
        write_report(args.output_dir)
        return
    if not args.model:
        raise SystemExit("--model is required unless --report-only is set")
    run_model(args)


if __name__ == "__main__":
    main()
