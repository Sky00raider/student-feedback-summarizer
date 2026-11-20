"""Command line: python -m app.cli <command>

  dataset                       write dataset/{train,val,test}.jsonl (+ the narrow "basic" version)
  train  [--r 8 --alpha 16]     train a LoRA adapter on distilgpt2, save models/adapter.pt
  merge                         merge the adapter into the weights, save models/merged/, check equality
  summarize FILE [--digest]     summarize a .csv/.txt/.json(l) of comments, print a digest
"""
import argparse
import json
import sys
import time
from pathlib import Path

from app import core


def cmd_dataset(a):
    for variety, sub in (("full", ""), ("basic", "basic")):
        d = core.make_dataset(a.train, a.val, a.test, seed=a.seed, variety=variety)
        out = core.DATASET_DIR / sub
        for split, rows in d.items():
            if sub and split == "test":
                continue                      # both versions are tested on the same full test split
            core.save_jsonl(rows, out / f"{split}.jsonl")
        print(f"{variety:5s} -> {out}/  " + "  ".join(f"{s}={len(r)}" for s, r in d.items()))
    diverse = core.load_diverse()
    if diverse:                               # stretch: template + varied hand-written comments
        template = core.load_jsonl(core.DATASET_DIR / "train.jsonl")
        train, dval = core.mix_datasets(template, diverse, a.mix_template)
        core.save_jsonl(train, core.DATASET_DIR / "mix" / "train.jsonl")
        core.save_jsonl(core.load_jsonl(core.DATASET_DIR / "val.jsonl") + dval, core.DATASET_DIR / "mix" / "val.jsonl")
        print(f"mix   -> {core.DATASET_DIR}/mix/  train={len(train)} ({min(a.mix_template, len(template))} template + "
              f"{len(train) - min(a.mix_template, len(template))} varied)  val=60 template + {len(dval)} varied")
    print(json.dumps(core.dataset_stats(core.load_jsonl(core.DATASET_DIR / "train.jsonl")), indent=1))


def cmd_train(a):
    train = core.load_jsonl(Path(a.data) / "train.jsonl")
    val = core.load_jsonl(Path(a.data) / "val.jsonl")
    cfg = core.TrainConfig(r=a.r, alpha=a.alpha, epochs=a.epochs, lr=a.lr, targets=tuple(a.targets.split(",")))
    model, tok = core.load_base(cfg.base)
    core.inject_lora(model, cfg.targets, cfg.r, cfg.alpha)
    lora, total = core.count_parameters(model)
    print(f"LoRA r={cfg.r} alpha={cfg.alpha:g} on {cfg.targets}: {lora:,} trainable of {total:,} ({lora / total:.3%})")
    hist = core.train_lora(model, tok, train, cfg, val)
    size = core.save_adapter(model, cfg, a.out, meta={"history": hist, "data": a.data, "n_train": len(train)})
    print(f"saved {a.out} ({size / 1024:.0f} KB) after {hist['seconds']:.0f}s; val loss per epoch {hist['val_loss']}")


def cmd_merge(a):
    import torch
    saved = core.read_adapter(a.adapter)
    lora_model, tok = core.load_base(saved["config"]["base"])
    core.apply_adapter(lora_model, saved)
    merged, _ = core.load_base(saved["config"]["base"])
    core.apply_adapter(merged, saved)
    n = core.merge_lora(merged)
    probe = [e.comment for e in core.load_jsonl(core.DATASET_DIR / "test.jsonl")[:16]]
    ids = torch.tensor([core.prompt_ids(tok, probe[0])])
    diff = core.max_logit_diff(lora_model, merged, ids)
    a_out, b_out = core.Summarizer(lora_model, tok).raw(probe), core.Summarizer(merged, tok).raw(probe)
    t_lora = core.time_generation(core.Summarizer(lora_model, tok), probe, 2)
    t_merged = core.time_generation(core.Summarizer(merged, tok), probe, 2)
    size = core.save_merged(merged, tok, a.out)
    print(f"merged {n} LoRA layers; max |logit difference| {diff:.2e}; "
          f"identical outputs on {sum(x == y for x, y in zip(a_out, b_out))}/{len(probe)} comments")
    print(f"latency: {t_lora:.0f} ms/comment with LoRA layers, {t_merged:.0f} ms/comment merged")
    print(f"wrote {a.out}/ ({size / 1e6:.0f} MB, a plain GPT-2 that loads without any LoRA code)")


def cmd_summarize(a):
    path = Path(a.file)
    comments = core.read_feedback(path.name, path.read_bytes())
    s = (core.Summarizer.from_merged(a.merged) if a.merged else core.Summarizer.from_adapter(a.adapter))
    t0 = time.time()
    results = s.summarize(comments, redact_pii=not a.keep_pii)
    if a.json:
        print(json.dumps([r.__dict__ for r in results], indent=1, ensure_ascii=False))
        return
    for r in results:
        flag = "" if r.valid else "  [unparsed]"
        print(f"{r.topic:<11} {r.sentiment:<8} {r.summary}{flag}")
    print(f"\n{len(results)} comments in {time.time() - t0:.1f}s")
    if a.digest:
        print()
        print(core.digest_markdown(core.build_digest(results)), end="")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--train", type=int, default=1200)
    d.add_argument("--val", type=int, default=60)
    d.add_argument("--test", type=int, default=120)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--mix-template", type=int, default=600, help="template examples in dataset/mix")
    t = sub.add_parser("train")
    t.add_argument("--data", default="dataset")
    t.add_argument("--r", type=int, default=8)
    t.add_argument("--alpha", type=float, default=16.0)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--targets", default="c_attn")
    t.add_argument("--out", default=str(core.DEFAULT_ADAPTER))
    m = sub.add_parser("merge")
    m.add_argument("--adapter", default=str(core.DEFAULT_ADAPTER))
    m.add_argument("--out", default="models/merged")
    s = sub.add_parser("summarize")
    s.add_argument("file")
    s.add_argument("--adapter", default=str(core.DEFAULT_ADAPTER))
    s.add_argument("--merged", help="load a merged model folder instead of base + adapter")
    s.add_argument("--digest", action="store_true")
    s.add_argument("--keep-pii", action="store_true", help="do not redact emails, phone numbers, IDs, names")
    s.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    {"dataset": cmd_dataset, "train": cmd_train, "merge": cmd_merge, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
