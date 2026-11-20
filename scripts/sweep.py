"""Stretch ideas 1 and 2: how LoRA rank, alpha and training data change summary quality.

Every run gets the same budget (400 training examples, 2 epochs, lr 2e-3) and is scored on the
mixed validation split (60 template + 32 varied) and on the independent freeform set. Resumable: finished runs in results/sweep.json are skipped.
Run: python scripts/sweep.py      (about 3-4 minutes per run on a laptop CPU)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import core  # noqa: E402

ALL = ("c_attn", "c_proj", "c_fc")
RUNS = [  # name, data folder, r, alpha, targets
    ("r=2 alpha=4", "dataset/mix", 2, 4, ALL),
    ("r=8 alpha=16", "dataset/mix", 8, 16, ALL),
    ("r=32 alpha=64", "dataset/mix", 32, 64, ALL),
    ("r=8 alpha=64", "dataset/mix", 8, 64, ALL),
    ("r=8 alpha=16, template data only", "dataset", 8, 16, ALL),
]
N_TRAIN, EPOCHS, LR = 400, 2, 2e-3


def main():
    out_json = ROOT / "results" / "sweep.json"
    done = json.loads(out_json.read_text()) if out_json.exists() else {}
    val = core.load_jsonl(ROOT / "dataset/mix/val.jsonl")
    free = core.load_jsonl(ROOT / "dataset/freeform.jsonl")
    for name, data, r, alpha, targets in RUNS:
        if name in done:
            continue
        train = core.load_jsonl(ROOT / data / "train.jsonl")[:N_TRAIN]   # mix is pre-shuffled: ~1/3 varied
        cfg = core.TrainConfig(r=r, alpha=alpha, targets=targets, epochs=EPOCHS, lr=LR)
        model, tok = core.load_base(cfg.base)
        core.inject_lora(model, targets, r, alpha)
        n_lora, _ = core.count_parameters(model)
        hist = core.train_lora(model, tok, train, cfg, val, log=lambda m: print(name, m, flush=True))
        core.merge_lora(model)
        s = core.Summarizer(model, tok)
        res = {"params": n_lora, "seconds": hist["seconds"], "val_loss": hist["val_loss"][-1]}
        for set_name, gold in (("val", val), ("freeform", free)):
            m = core.evaluate(s.summarize([g.comment for g in gold], redact_pii=False), gold)
            res[set_name] = {k: m[k] for k in ("topic_acc", "sentiment_acc", "rougeL", "valid_format", "supported_words")}
        done[name] = res
        print(name, res, flush=True)
        out_json.parent.mkdir(exist_ok=True)
        out_json.write_text(json.dumps(done, indent=1), encoding="utf-8")
    write_md(done)


def write_md(done):
    lines = ["# LoRA sweep", "",
             f"Same budget for every run: first {N_TRAIN} training examples, {EPOCHS} epochs, lr {LR:g}, distilgpt2. "
             "Scores: val = 60 template + 32 varied validation comments, free = 48 independent freeform comments.", "",
             "| Run | Trainable params | Train s | Val loss | Val topic % | Val sent. % | Val ROUGE-L | "
             "Free topic % | Free sent. % | Free ROUGE-L | Free words in comment % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in done.items():
        v, f = r["val"], r["freeform"]
        lines.append(f"| {name} | {r['params']:,} | {r['seconds']:.0f} | {r['val_loss']:.3f} | {100 * v['topic_acc']:.1f} | "
                     f"{100 * v['sentiment_acc']:.1f} | {100 * v['rougeL']:.1f} | {100 * f['topic_acc']:.1f} | "
                     f"{100 * f['sentiment_acc']:.1f} | {100 * f['rougeL']:.1f} | {100 * f['supported_words']:.1f} |")
    (ROOT / "results" / "sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote results/sweep.md")


if __name__ == "__main__":
    main()
