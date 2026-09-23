"""Run a Tower checkpoint over eval/data/eval_data.jsonl with greedy decoding
(matches the paper's main-table setup, Appendix F.8) and write hypotheses out.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="path or HF id of the model to evaluate")
p.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data", "eval_data.jsonl"))
p.add_argument("--out", required=True, help="output JSONL path")
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--max_new_tokens", type=int, default=512)
p.add_argument("--testsets", default="", help="comma-separated filter, e.g. flores200,wmt23")
a = p.parse_args()

rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
if a.testsets:
    keep = set(a.testsets.split(","))
    rows = [r for r in rows if r["testset"] in keep]
print(f"{len(rows)} examples to translate", flush=True)

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa", device_map="cuda")
model.eval()

im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
eos_ids = [tok.eos_token_id] + ([im_end_id] if im_end_id != tok.unk_token_id else [])

# Longest prompts first within each batch bucket -> less padding waste, stable memory.
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]))

os.makedirs(os.path.dirname(a.out), exist_ok=True)
results = [None] * len(rows)
with torch.no_grad():
    for start in range(0, len(order), a.batch_size):
        idxs = order[start:start + a.batch_size]
        prompts = [rows[i]["prompt"] for i in idxs]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                             num_beams=1, eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idxs):
            new_tokens = gen[j][enc["input_ids"].shape[1]:]
            text = tok.decode(new_tokens, skip_special_tokens=True).strip()
            results[i] = text
        if start % (a.batch_size * 20) == 0:
            print(f"{start}/{len(order)}", flush=True)

with open(a.out, "w", encoding="utf-8") as f:
    for row, hyp in zip(rows, results):
        out_row = dict(row)
        out_row["hyp"] = hyp
        f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
print("done ->", a.out)
