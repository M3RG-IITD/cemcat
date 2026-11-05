"""Download the mineralogy-benchmark checkpoints into the Hugging Face cache.

Gemma is gated, so HF_TOKEN must be set for that repo.

Example:
  python scripts/download_mineralogy_models.py --check
  python scripts/download_mineralogy_models.py --models qwen25-7b gemma3-4b ministral-8b
  python scripts/download_mineralogy_models.py --models llamat-3-chat llamat-2-chat
"""

import argparse
import os

MODELS = [
    ("qwen25-0.5b", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("gemma3-270m", "google/gemma-3-270m-it"),
    ("ministral-3b", "mistralai/Ministral-3-3B-Instruct-2512"),
    ("qwen25-7b", "Qwen/Qwen2.5-7B-Instruct"),
    ("gemma3-4b", "google/gemma-3-4b-it"),
    ("ministral-8b", "mistralai/Ministral-8B-Instruct-2410"),
    ("llamat-2-chat", "m3rg-iitd/llamat-2-chat"),
    ("llamat-3-chat", "m3rg-iitd/llamat-3-chat"),
]

def default_cache_root():
    if os.environ.get("CACHE_ROOT"):
        return os.environ["CACHE_ROOT"]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.dirname(hf_home.rstrip(os.sep))
    if os.environ.get("XDG_CACHE_HOME"):
        return os.environ["XDG_CACHE_HOME"]
    return os.path.join(os.path.expanduser("~"), ".cache")


def parse_args():
    parser = argparse.ArgumentParser(description="Prefetch mineralogy benchmark weights")
    parser.add_argument(
        "--cache-root",
        default=default_cache_root(),
        help="Cache root. Weights go in <root>/huggingface/hub unless HF_HOME is already set.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report which checkpoints are already cached",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Keys to fetch. Defaults to every checkpoint.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def configure_cache(cache_root):
    hf_home = os.environ.get("HF_HOME") or os.path.join(cache_root, "huggingface")
    hub_cache = os.environ.get("HF_HUB_CACHE") or os.path.join(hf_home, "hub")
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = hub_cache
    os.environ["TRANSFORMERS_CACHE"] = os.environ.get(
        "TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers")
    )
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    # Use the Xet client when it is installed. A previous disable left large
    # files on the slow HTTP fallback.
    os.environ.pop("HF_HUB_DISABLE_XET", None)
    os.makedirs(hub_cache, exist_ok=True)
    return hf_home


def cache_dir(model_id):
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
        "hub",
    )
    return os.path.join(hub, f"models--{model_id.replace('/', '--')}")


def cached_size_gib(model_id):
    path = cache_dir(model_id)
    if not os.path.isdir(path):
        return 0.0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            if os.path.islink(full):
                continue
            try:
                total += os.path.getsize(full)
            except OSError:
                pass
    return total / (1024**3)


# vLLM loads this checkpoint with --load-format mistral, which reads
# consolidated.safetensors. model.safetensors is the same weights again.
IGNORE_PATTERNS = {
    "ministral-3b": ["model.safetensors"],
}


def selected_models(keys):
    known = dict(MODELS)
    if not keys:
        return list(MODELS)
    missing = [key for key in keys if key not in known]
    if missing:
        raise SystemExit(f"Unknown model key(s): {', '.join(missing)}")
    return [(key, known[key]) for key in keys]


def main():
    args = parse_args()
    hf_home = configure_cache(args.cache_root)
    models = selected_models(args.models)
    print(f"HF_HOME: {hf_home}")
    print(f"{'key':16s} {'cached GiB':>11s}  checkpoint")
    for key, model_id in models:
        size = cached_size_gib(model_id)
        print(f"{key:16s} {size:11.2f}  {model_id}")

    if args.check:
        return

    import time

    from huggingface_hub import snapshot_download

    failed = []
    for key, model_id in models:
        print(f"\n=== {key} -> {model_id}")
        ignore = ["*.pth", "*.msgpack", "*.h5", "original/*", *IGNORE_PATTERNS.get(key, [])]
        last_error = None
        for attempt in range(1, args.retries + 1):
            try:
                snapshot_download(
                    repo_id=model_id,
                    max_workers=args.workers,
                    ignore_patterns=ignore,
                )
                print(f"  done ({cached_size_gib(model_id):.2f} GiB)")
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - Hub transfers drop mid-file
                last_error = exc
                print(f"  attempt {attempt}/{args.retries} FAILED: {type(exc).__name__}: {exc}")
                if attempt < args.retries:
                    time.sleep(min(2**attempt, 30))
        if last_error is not None:
            failed.append(key)
    if failed:
        raise SystemExit(f"Download failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
