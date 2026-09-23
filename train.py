"""Full fine-tune of TowerBase on TowerBlocks (ChatML, loss on assistant turns only)."""
import argparse, glob, os
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)

p = argparse.ArgumentParser()
p.add_argument("--model", default=os.path.expanduser("~/models/TowerBase-7B"))
p.add_argument("--data", default=os.path.expanduser("~/data/TowerBlocks/data"))
p.add_argument("--out", required=True)
p.add_argument("--tasks", default="", help="comma-separated task filter, empty = all")
p.add_argument("--max_samples", type=int, default=0)
p.add_argument("--max_len", type=int, default=2048)
p.add_argument("--epochs", type=float, default=4)
p.add_argument("--lr", type=float, default=7e-6)
p.add_argument("--global_bs", type=int, default=256)
p.add_argument("--micro_bs", type=int, default=4)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--dry_run", action="store_true", help="only tokenize + print stats")
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"], "pad_token": "<pad>"})
BOS = tok.bos_token

def encode(ex):
    text, spans = BOS, []
    for turn in ex["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        head = f"<|im_start|>{role}\n"
        body = f"{turn['value']}<|im_end|>"
        text += head
        if role == "assistant":
            spans.append((len(text), len(text) + len(body)))
        text += body + "\n"
    enc = tok(text, add_special_tokens=False, truncation=True, max_length=a.max_len,
              return_offsets_mapping=True)
    labels = [i if any(s <= o[0] < e for s, e in spans) else -100
              for i, o in zip(enc["input_ids"], enc["offset_mapping"])]
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

ds = load_dataset("parquet", data_files=sorted(glob.glob(f"{a.data}/*.parquet")))["train"]
if a.tasks:
    keep = set(a.tasks.split(","))
    ds = ds.filter(lambda t: t in keep, input_columns="task", num_proc=8)
ds = ds.shuffle(seed=a.seed)
if a.max_samples:
    ds = ds.select(range(min(a.max_samples, len(ds))))
ds = ds.map(encode, remove_columns=ds.column_names, num_proc=16, desc="tokenize")
ds = ds.filter(lambda l: any(x != -100 for x in l), input_columns="labels", num_proc=8)
print(f"examples={len(ds)} tokens={sum(len(x) for x in ds['input_ids']):,}", flush=True)
if a.dry_run:
    ex = ds[0]
    print(tok.decode(ex["input_ids"])[:600])
    print("--- trained on ---")
    print(tok.decode([i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100])[:400])
    raise SystemExit

model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa")
model.resize_token_embeddings(len(tok), pad_to_multiple_of=64)
model.config.pad_token_id = tok.pad_token_id
model.config.eos_token_id = tok.convert_tokens_to_ids("<|im_end|>")  # paper: EOS overwritten with <|im_end|>
model.config.use_cache = False

world = int(os.environ.get("WORLD_SIZE", 1))
accum = max(1, a.global_bs // (a.micro_bs * world))
args = TrainingArguments(
    output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
    lr_scheduler_type="cosine", warmup_steps=500, weight_decay=0.01,
    per_device_train_batch_size=a.micro_bs, gradient_accumulation_steps=accum,
    bf16=True, gradient_checkpointing=True, group_by_length=True,
    fsdp="full_shard auto_wrap",
    fsdp_config={"transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"], "use_orig_params": True,
                 "sync_module_states": True, "cpu_ram_efficient_loading": True,
                 "activation_checkpointing": False},
    logging_steps=10, save_strategy="steps", save_steps=500, save_total_limit=2,
    save_only_model=True, report_to="none", seed=a.seed, dataloader_num_workers=4,
    remove_unused_columns=False,
)
trainer = Trainer(model=model, args=args, train_dataset=ds,
                  data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100,
                                                       pad_to_multiple_of=8))
trainer.train(resume_from_checkpoint=bool(glob.glob(f"{a.out}/checkpoint-*")))
trainer.save_model(a.out + "/final")
tok.save_pretrained(a.out + "/final")
