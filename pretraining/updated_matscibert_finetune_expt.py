import torch
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import math
import os
from pathlib import Path
from tqdm import tqdm

from argparse import ArgumentParser

from transformers import (
    AutoConfig,
    BertForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,  # CHANGE 1: replaced DataCollatorForWholeWordMask
    Trainer,
    TrainingArguments,
    set_seed,
)

from joblib import Parallel, delayed  # CHANGE 2: removed duplicate/unused imports (multiprocessing, tqdm imported twice)

class WholeWordMaskMLMCollator(DataCollatorForLanguageModeling):
    """
    Thin fix: extracts offset_mapping from features before tokenizer.pad()
    drops it, then passes it properly to the standard torch_mask_tokens().
    All masking logic is still the parent class (standard library).
    """
    def torch_call(self, examples):
        if self.seed and self.generator is None:
            self.create_rng()

        # ── Save offset_mapping before pad() drops it ────────────────────
        offset_mappings = None
        if isinstance(examples[0], dict) and "offset_mapping" in examples[0]:
            offset_mappings = [e.pop("offset_mapping") for e in examples]

        # ── Standard padding / batching (identical to parent) ────────────
        if isinstance(examples[0], dict):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            from transformers.data.data_collator import _torch_collate_batch
            batch = {
                "input_ids": _torch_collate_batch(
                    examples, self.tokenizer,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                )
            }

        special_tokens_mask = batch.pop("special_tokens_mask", None)

        # ── Re-stack offset_mapping into [batch, seq_len, 2] tensor ──────
        if offset_mappings is not None:
            offset_mapping = torch.stack([
                t if isinstance(t, torch.Tensor) else torch.tensor(t)
                for t in offset_mappings
            ])
        else:
            offset_mapping = None

        # ── Masking — 100% standard library from here ────────────────────
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"],
                special_tokens_mask=special_tokens_mask,
                offset_mapping=offset_mapping,
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        return batch



def run_inference(pycommand=None):
    print('Running: ', pycommand)
    os.system(pycommand)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)


def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


parser = ArgumentParser()
parser.add_argument('--train_file', required=True, type=str)
parser.add_argument('--val_file', required=True, type=str)
parser.add_argument('--model_save_dir', required=True, type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--eval_batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-4, type=float)

# NEW ADDED
parser.add_argument('--debug', action='store_true', help='Run on small subset for testing')


args = parser.parse_args()

model_revision = 'main'
model_name = '/home/scai/msr/aiy257591/models/m3rg-iitd/matscibert'
cache_dir = ensure_dir(args.cache_dir) if args.cache_dir else None
output_dir = ensure_dir(args.model_save_dir)

assert os.path.exists(args.train_file)
assert os.path.exists(args.val_file)

SEED = 42
set_seed(SEED)

config_kwargs = {
    'cache_dir': cache_dir,
    'revision': model_revision,
    'use_auth_token': None,
}
config = AutoConfig.from_pretrained(model_name, **config_kwargs)

tokenizer_kwargs = {
    'cache_dir': cache_dir,
    'use_fast': True,
    'revision': model_revision,
    'use_auth_token': None,
}
tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

max_seq_length = 512

start_tok = tokenizer.convert_tokens_to_ids('[CLS]')
sep_tok = tokenizer.convert_tokens_to_ids('[SEP]')
pad_tok = tokenizer.convert_tokens_to_ids('[PAD]')



# NEW MODIFIED

def full_sent_tokenize(file_name):
    with open(file_name, 'r') as f:
        sents = f.read().strip().split('\n')

    num_cores = -1

    def tokme(s):
        out = tokenizer(s, padding=False, truncation=False,
                        return_offsets_mapping=True)
        return out['input_ids'], out['offset_mapping']

    results = Parallel(n_jobs=num_cores)(
        delayed(tokme)(s) for s in tqdm(sents)
    )
    tok_sents    = [r[0] for r in results]
    offset_sents = [r[1] for r in results]

    # strip [CLS] (index 0) from both
    for s in tok_sents:
        s.pop(0)
    for s in offset_sents:
        s.pop(0)

    res         = [[]]
    res_offsets = [[]]
    l_curr = 0

    for s, off in zip(tok_sents, offset_sents):
        l_s = len(s)
        idx = 0
        while idx < l_s - 1:
            if l_curr == 0:
                res[-1].append(start_tok)
                res_offsets[-1].append((0, 0))   # CLS — special token
                l_curr = 1
            s_end = min(l_s, idx + max_seq_length - l_curr) - 1
            res[-1].extend(s[idx:s_end] + [sep_tok])
            res_offsets[-1].extend(off[idx:s_end] + [(0, 0)])  # SEP — special token
            idx = s_end
            if len(res[-1]) == max_seq_length:
                res.append([])
                res_offsets.append([])
            l_curr = len(res[-1])

    for s in res[:-1]:
        assert s[0] == start_tok and s[-1] == sep_tok
        assert len(s) == max_seq_length

    attention_mask   = []
    offset_mapping   = []

    for i, s in enumerate(res):
        pad_len = max_seq_length - len(s)
        attention_mask.append([1] * len(s) + [0] * pad_len)
        # pad offsets with (0, 0) for PAD tokens
        offset_mapping.append(res_offsets[i] + [(0, 0)] * pad_len)

    return {
        'input_ids':      res,
        'attention_mask':  attention_mask,
        'offset_mapping':  offset_mapping,
    }


class MSC_Dataset(torch.utils.data.Dataset):
    def __init__(self, inp):
        self.inp = inp

    def __getitem__(self, idx):
        return {
            'input_ids':      torch.tensor(self.inp['input_ids'][idx]),
            'attention_mask':  torch.tensor(self.inp['attention_mask'][idx]),
            'offset_mapping':  torch.tensor(self.inp['offset_mapping'][idx]),  # shape [seq_len, 2]
        }

    def __len__(self):
        return len(self.inp['input_ids'])


train_dataset = MSC_Dataset(full_sent_tokenize(args.train_file))
eval_dataset = MSC_Dataset(full_sent_tokenize(args.val_file))

# NEW ADDED
if args.debug:
    train_dataset = torch.utils.data.Subset(train_dataset, range(min(256, len(train_dataset))))
    eval_dataset  = torch.utils.data.Subset(eval_dataset,  range(min(64,  len(eval_dataset))))
    print(f"[DEBUG] Subsetted to {len(train_dataset)} train / {len(eval_dataset)} eval samples")

print(len(train_dataset), len(eval_dataset))


model = BertForMaskedLM.from_pretrained(
    model_name,
    from_tf=False,
    config=config,
    cache_dir=cache_dir,
    revision=model_revision,
)
model.resize_token_embeddings(len(tokenizer))

# CHANGE 4: replaced DataCollatorForWholeWordMask with DataCollatorForLanguageModeling
# whole_word_mask=True preserves the exact same behavior

# REVERT_BACK: BELOW CODE giving error 

data_collator = WholeWordMaskMLMCollator(
    tokenizer=tokenizer,
    mlm_probability=0.15,
    whole_word_mask=True,
)


print("\n===== MLM COLLATOR CONFIG =====")
print("whole_word_mask     :", data_collator.whole_word_mask)
print("mlm_probability     :", data_collator.mlm_probability)
print("mask_replace_prob   :", data_collator.mask_replace_prob)
print("random_replace_prob :", data_collator.random_replace_prob)
print("===============================\n")



# CHANGE 5: batch size now derived from TARGET_BATCH_SIZE to preserve paper's
# effective batch size of 256 regardless of number of GPUs
TARGET_BATCH_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = 8
world_size = int(os.environ.get("WORLD_SIZE", 1))
effective_batch_per_step = world_size * GRADIENT_ACCUMULATION_STEPS

if TARGET_BATCH_SIZE % effective_batch_per_step != 0:
    raise ValueError(
        f"Cannot achieve batch size {TARGET_BATCH_SIZE} with "
        f"{world_size} GPUs and gradient accumulation "
        f"{GRADIENT_ACCUMULATION_STEPS}."
    )

per_device_batch_size = TARGET_BATCH_SIZE // effective_batch_per_step

# CHANGE 6: warmup_ratio replaced with manually computed warmup_steps
# because transformers v5.15.0 dropped warmup_ratio from TrainingArguments
num_update_steps_per_epoch = math.ceil(len(train_dataset) / TARGET_BATCH_SIZE)
total_train_steps = num_update_steps_per_epoch * args.epochs
warmup_steps = math.ceil(0.048 * total_train_steps)

print(f"GPUs: {world_size}")
print(f"Per-GPU batch size: {per_device_batch_size}")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {per_device_batch_size * world_size * GRADIENT_ACCUMULATION_STEPS}")
print(f"Updates/epoch: {num_update_steps_per_epoch}")
print(f"Total optimizer steps: {total_train_steps}")
print(f"Warmup steps: {warmup_steps}")


# NEW ADDED
eval_steps  = 10 if args.debug else 500
save_steps  = 10 if args.debug else 500
epochs      = 2  if args.debug else args.epochs

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    eval_strategy='steps',
    eval_steps= eval_steps,         # CHANGE 7: was 5, far too frequent for a large corpus
    save_strategy='steps',
    save_steps= save_steps,         # CHANGE 7: same as above
    # save_total_limit=3,     # CHANGE 7: keep only 3 best checkpoints to save disk space
    load_best_model_at_end=True,
    warmup_steps=warmup_steps,
    learning_rate=args.lr,
    weight_decay=1e-2,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    max_grad_norm=0.0,      # NOTE: 0.0 = no gradient clipping; intentional per MatSciBERT setup
    num_train_epochs= epochs,
    seed=SEED,
    remove_unused_columns=False,   # NEW ADDED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,  # CHANGE 8: replaced deprecated tokenizer= argument
    data_collator=data_collator,
)

resume = None if len(os.listdir(output_dir)) == 0 else True
train_res = trainer.train(resume_from_checkpoint=resume)
print(train_res)

train_output = trainer.evaluate(train_dataset)
eval_output = trainer.evaluate()

print(train_output)
print(eval_output)