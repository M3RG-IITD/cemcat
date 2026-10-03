import hashlib
import os
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.trainer_utils import get_last_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────────────────
class WholeWordMaskMLMCollator(DataCollatorForLanguageModeling):
    """
    Pulls offset_mapping out of the features before tokenizer.pad() drops it,
    then hands it to the standard torch_mask_tokens() for whole-word masking.
    All masking logic is the parent class.
    """

    def torch_call(self, examples):
        if self.seed and self.generator is None:
            self.create_rng()

        offset_mappings = [e.pop("offset_mapping") for e in examples]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, examples, return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        offset_mapping = torch.stack(offset_mappings)  # [batch, seq_len, 2]

        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"],
            special_tokens_mask=special_tokens_mask,
            offset_mapping=offset_mapping,
        )
        return batch


# ─────────────────────────────────────────────────────────────────────────────
# Args / setup
# ─────────────────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument('--train_file', required=True, type=str)
parser.add_argument('--val_file', required=True, type=str)
parser.add_argument('--model_save_dir', required=True, type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--eval_batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
args = parser.parse_args()

# Knows rank / world size under torchrun. The long timeout stops the other
# ranks from crashing (default: 30 min) while rank 0 tokenises the corpus
# the first time.
DIST_TIMEOUT = timedelta(hours=6)
state = PartialState(timeout=DIST_TIMEOUT)


def log(*a):
    if state.is_main_process:
        print(*a, flush=True)


def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


model_name = '/home/scai/msr/aiy257591/models/m3rg-iitd/matscibert'
cache_dir = ensure_dir(args.cache_dir) if args.cache_dir else None
output_dir = ensure_dir(args.model_save_dir)
tok_cache_root = ensure_dir(os.path.join(cache_dir or output_dir, 'tokenized'))

assert os.path.exists(args.train_file)
assert os.path.exists(args.val_file)

SEED = 42
set_seed(SEED)

config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)

max_seq_length = 512
start_tok = tokenizer.convert_tokens_to_ids('[CLS]')
sep_tok = tokenizer.convert_tokens_to_ids('[SEP]')
pad_tok = tokenizer.convert_tokens_to_ids('[PAD]')

TOKENIZE_CHUNK = 10_000  # sentences per fast-tokenizer batch call


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation → fixed 512-token blocks (same packing as original MatSciBERT)
# ─────────────────────────────────────────────────────────────────────────────
def build_blocks(file_name):
    with open(file_name, 'r') as f:
        sents = f.read().strip().split('\n')

    blocks_ids, blocks_off = [], []
    cur_ids, cur_off = [], []

    def flush(ids, off):
        pad_len = max_seq_length - len(ids)
        blocks_ids.append(np.asarray(ids + [pad_tok] * pad_len, dtype=np.int32))
        blocks_off.append(np.asarray(off + [(0, 0)] * pad_len, dtype=np.int32))

    for start in tqdm(range(0, len(sents), TOKENIZE_CHUNK), desc=f'Tokenizing {Path(file_name).name}'):
        enc = tokenizer(sents[start:start + TOKENIZE_CHUNK], padding=False,
                        truncation=False, return_offsets_mapping=True)
        for s, off in zip(enc['input_ids'], enc['offset_mapping']):
            s, off = s[1:], off[1:]          # drop [CLS]; sentence still ends in [SEP]
            l_s, idx = len(s), 0
            while idx < l_s - 1:
                if not cur_ids:
                    cur_ids.append(start_tok)
                    cur_off.append((0, 0))
                s_end = min(l_s, idx + max_seq_length - len(cur_ids)) - 1
                cur_ids.extend(s[idx:s_end] + [sep_tok])
                cur_off.extend(off[idx:s_end] + [(0, 0)])
                idx = s_end
                if len(cur_ids) == max_seq_length:
                    flush(cur_ids, cur_off)
                    cur_ids, cur_off = [], []

    if cur_ids:                               # last partial block → pad to 512
        flush(cur_ids, cur_off)

    return np.stack(blocks_ids), np.stack(blocks_off)


def load_or_build(file_name):
    """Rank 0 tokenises once and caches to .npy; all ranks memory-map the cache."""
    st = os.stat(file_name)
    key = hashlib.md5(
        f'{os.path.abspath(file_name)}|{st.st_size}|{st.st_mtime_ns}|{max_seq_length}|{model_name}'.encode()
    ).hexdigest()[:16]
    d = Path(tok_cache_root) / f'{Path(file_name).stem}_{key}'
    ids_path, off_path = d / 'input_ids.npy', d / 'offset_mapping.npy'

    with state.main_process_first():
        if not (ids_path.exists() and off_path.exists()):
            d.mkdir(parents=True, exist_ok=True)
            ids, off = build_blocks(file_name)
            np.save(d / 'input_ids.tmp.npy', ids)
            np.save(d / 'offset_mapping.tmp.npy', off)
            os.replace(d / 'input_ids.tmp.npy', ids_path)
            os.replace(d / 'offset_mapping.tmp.npy', off_path)
            del ids, off

    return np.load(ids_path, mmap_mode='r'), np.load(off_path, mmap_mode='r')


special_ids = torch.tensor(tokenizer.all_special_ids)


class MSC_Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, offset_mapping):
        self.input_ids = input_ids
        self.offset_mapping = offset_mapping

    def __getitem__(self, idx):
        ids = torch.from_numpy(np.array(self.input_ids[idx], dtype=np.int64))
        return {
            'input_ids': ids,
            'attention_mask': (ids != pad_tok).long(),
            'special_tokens_mask': torch.isin(ids, special_ids).long(),
            'offset_mapping': torch.from_numpy(np.array(self.offset_mapping[idx], dtype=np.int64)),
        }

    def __len__(self):
        return len(self.input_ids)


train_dataset = MSC_Dataset(*load_or_build(args.train_file))
eval_dataset = MSC_Dataset(*load_or_build(args.val_file))
log(len(train_dataset), len(eval_dataset))


# ─────────────────────────────────────────────────────────────────────────────
# Model / collator
# ─────────────────────────────────────────────────────────────────────────────
# Keep master weights in fp32 (v5 defaults to dtype="auto"); bf16 below is autocast only.
model = BertForMaskedLM.from_pretrained(model_name, config=config, cache_dir=cache_dir, dtype=torch.float32)
model.resize_token_embeddings(len(tokenizer))

# BertForMaskedLM.forward has **kwargs, so Trainer (v5) assumes the model normalises
# the loss by num_items_in_batch itself. It doesn't (plain mean CrossEntropy), which
# makes Trainer skip the /grad_accum division and multiply the loss by world_size:
# logged train loss becomes ~ (grad_accum x n_gpus) too big, eval loss ~ n_gpus too big.
# This flag restores the correct scaling.
model.accepts_loss_kwargs = False

data_collator = WholeWordMaskMLMCollator(
    tokenizer=tokenizer,
    mlm_probability=0.15,
    whole_word_mask=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Batch size / schedule  (effective batch 256, as in the paper)
# ─────────────────────────────────────────────────────────────────────────────
TARGET_BATCH_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = 8
world_size = int(os.environ.get('WORLD_SIZE', 1))

if world_size == 1 and torch.cuda.device_count() > 1:
    raise RuntimeError(
        f'{torch.cuda.device_count()} GPUs visible but WORLD_SIZE=1. Launch with '
        f'`torchrun --nproc_per_node=N`; plain `python` falls back to DataParallel '
        f'and the effective batch size would be {torch.cuda.device_count()}x too large.'
    )

effective_batch_per_step = world_size * GRADIENT_ACCUMULATION_STEPS
if TARGET_BATCH_SIZE % effective_batch_per_step != 0:
    raise ValueError(
        f'Cannot achieve batch size {TARGET_BATCH_SIZE} with {world_size} GPUs '
        f'and gradient accumulation {GRADIENT_ACCUMULATION_STEPS}.'
    )
per_device_batch_size = TARGET_BATCH_SIZE // effective_batch_per_step

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

log(f'GPUs: {world_size}')
log(f'Per-GPU batch size: {per_device_batch_size}')
log(f'Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}')
log(f'Effective batch size: {per_device_batch_size * world_size * GRADIENT_ACCUMULATION_STEPS}')
log(f'bf16 mixed precision: {use_bf16}')


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    eval_strategy='steps',
    eval_steps=50,
    save_strategy='steps',
    save_steps=50,
    # save_total_limit=3,
    load_best_model_at_end=True,
    logging_steps=50,
    logging_first_step=True,
    warmup_steps=0.048,             # float < 1 = ratio of total steps (v5 replacement for warmup_ratio)
    learning_rate=args.lr,
    weight_decay=1e-2,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    max_grad_norm=0.0,              # no gradient clipping (MatSciBERT setup)
    num_train_epochs=args.epochs,
    seed=SEED,
    bf16=use_bf16,                  # mixed precision on Ampere+ GPUs, fp32 otherwise
    remove_unused_columns=False,    # keep offset_mapping for the collator
    ddp_find_unused_parameters=False,
    ddp_timeout=int(DIST_TIMEOUT.total_seconds()),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

last_checkpoint = get_last_checkpoint(output_dir)
log(f'Resuming from: {last_checkpoint}' if last_checkpoint else 'Starting fresh')

train_res = trainer.train(resume_from_checkpoint=last_checkpoint)
log(train_res)

train_output = trainer.evaluate(train_dataset, metric_key_prefix='train')
eval_output = trainer.evaluate()
log(train_output)
log(eval_output)