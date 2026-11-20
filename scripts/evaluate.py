"""Milestone 4 + 5 evaluation: baselines vs the LoRA adapter vs the merged model.

Run: python scripts/evaluate.py [--adapter models/adapter.pt]
Writes results/results.md, results/results.json and results/samples.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import core  # noqa: E402

FEW_SHOT_IDS = (0, 1, 2)          # training examples shown to the base model in its prompt


def few_shot_summarizer(model, tok, shots):
    """The base model with no adapter, prompted with three solved examples."""
    prefix = "".join(f"{core.PROMPT_PREFIX}{e.comment}{core.PROMPT_SUFFIX}{core.target_text(e)}\n\n" for e in shots)
    pre = tok.encode(prefix)

    class FewShot(core.Summarizer):
        def raw(self, comments):
            prompts = [pre + core.prompt_ids(tok, c) for c in comments]
            return core.generate(self.model, self.tok, prompts, self.max_new_tokens, self.batch_size)
    return FewShot(model, tok)


def fmt(v):
    return f"{100 * v:.1f}" if isinstance(v, float) and v <= 1.0 else f"{v:.2f}" if isinstance(v, float) else str(v)


def table(rows: dict[str, dict], cols: list[tuple[str, str]]) -> str:
    out = ["| System | " + " | ".join(c for _, c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for name, m in rows.items():
        out.append(f"| {name} | " + " | ".join(fmt(m[k]) for k, _ in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(core.DEFAULT_ADAPTER))
    a = ap.parse_args()
    sets = {"synthetic test (held-out phrasings + courses)": core.load_jsonl(ROOT / "dataset/test.jsonl"),
            "freeform (48 comments written independently)": core.load_jsonl(ROOT / "dataset/freeform.jsonl")}
    train = core.load_jsonl(ROOT / "dataset/train.jsonl")
    saved = core.read_adapter(a.adapter)
    cfg = saved["config"]

    base, tok = core.load_base(cfg["base"])
    lora_model, _ = core.load_base(cfg["base"])
    core.apply_adapter(lora_model, saved)
    merged_model, _ = core.load_base(cfg["base"])
    core.apply_adapter(merged_model, saved)
    n_lora, n_total = core.count_parameters(merged_model)
    n_merged_layers = core.merge_lora(merged_model)

    systems = {
        "Keyword baseline (no model)": None,
        "distilgpt2, 3-shot prompt, no LoRA": few_shot_summarizer(base, tok, [train[i] for i in FEW_SHOT_IDS]),
        "distilgpt2 + LoRA (unmerged)": core.Summarizer(lora_model, tok),
        "distilgpt2 + LoRA, merged": core.Summarizer(merged_model, tok),
    }
    results, samples, timings = {}, {}, {}
    for set_name, gold in sets.items():
        comments = [g.comment for g in gold]
        results[set_name], samples[set_name] = {}, {}
        for name, s in systems.items():
            t0 = time.perf_counter()
            preds = [core.keyword_baseline(c) for c in comments] if s is None else s.summarize(comments, redact_pii=False)
            timings.setdefault(name, []).append(1000 * (time.perf_counter() - t0) / len(comments))
            results[set_name][name] = core.evaluate(preds, gold)
            samples[set_name][name] = preds
            print(set_name, "|", name, {k: round(v, 3) for k, v in results[set_name][name].items()}, flush=True)
        results[set_name]["Reference summaries (upper bound)"] = core.evaluate(core.gold_as_predictions(gold), gold)

    # merge checks
    ids = torch.tensor([core.prompt_ids(tok, sets[next(iter(sets))][0].comment)])
    diff = core.max_logit_diff(lora_model, merged_model, ids)
    same = {k: sum(x.raw == y.raw for x, y in zip(v["distilgpt2 + LoRA (unmerged)"], v["distilgpt2 + LoRA, merged"]))
            for k, v in samples.items()}
    probe = [g.comment for g in sets[next(iter(sets))][:32]]
    speed = {"unmerged": core.time_generation(systems["distilgpt2 + LoRA (unmerged)"], probe, 3),
             "merged": core.time_generation(systems["distilgpt2 + LoRA, merged"], probe, 3)}
    adapter_kb = Path(a.adapter).stat().st_size / 1024
    merge = {"layers_merged": n_merged_layers, "max_logit_diff": diff, "identical_outputs": same,
             "ms_per_comment": speed, "adapter_kb": adapter_kb, "lora_params": n_lora, "base_params": n_total,
             "config": {k: cfg[k] for k in ("r", "alpha", "targets", "lr", "epochs")},
             "train_seconds": saved["meta"].get("history", {}).get("seconds"), "n_train": saved["meta"].get("n_train")}
    print(merge)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps({"metrics": results, "merge": merge}, indent=1), encoding="utf-8")
    cols = [("topic_acc", "Topic acc %"), ("sentiment_acc", "Sentiment acc %"), ("issue_acc", "Issue-vs-praise acc %"), ("rouge1", "ROUGE-1"),
            ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L"), ("valid_format", "Well-formed %"),
            ("supported_words", "Words found in comment %"), ("summary_words", "Words / summary")]
    md = ["# Results", "", f"Adapter: `{Path(a.adapter).as_posix()}`, r={cfg['r']}, alpha={cfg['alpha']:g}, targets "
          f"{', '.join(cfg['targets'])}, {merge['n_train']} training examples, {cfg['epochs']} epochs, "
          f"trained in {merge['train_seconds']:.0f} s on a laptop CPU.", ""]
    for set_name in sets:
        md += [f"## {set_name}", "", table(results[set_name], cols), ""]
    md += ["ROUGE is F1 x 100 against the reference summary. 'Words found in comment' = share of the summary's "
           "content words that appear in the original comment (a hallucination proxy); the reference summaries "
           "paraphrase, so they score below 100 too. 'Issue-vs-praise' counts negative and mixed as one class: "
           "that is all the digest needs to decide whether a comment is an action item.", "",
           "## Merging (milestone 5)", "",
           f"- {n_merged_layers} LoRA layers merged into the weights (W' = W + alpha/r * B @ A).",
           f"- Max |logit difference| merged vs unmerged: {diff:.1e}.",
           "- Identical generated text: " + ", ".join(f"{v}/{len(sets[k])} on {k.split(' (')[0]}" for k, v in same.items()) + ".",
           f"- Latency (batch of 8, greedy, CPU): {speed['unmerged']:.0f} ms/comment unmerged, "
           f"{speed['merged']:.0f} ms/comment merged.",
           f"- Adapter file {adapter_kb:.0f} KB ({n_lora:,} trainable parameters, "
           f"{n_lora / n_total:.2%} of {n_total:,}); merged model = a plain distilgpt2 checkpoint.", ""]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")

    sm = ["# Sample outputs", ""]
    for set_name, gold in sets.items():
        sm += [f"## {set_name}", ""]
        for i in range(0, len(gold), max(1, len(gold) // 8))[:8]:
            g = gold[i]
            sm += [f"**Comment:** {g.comment}", "", f"- Reference: `{g.topic} | {g.sentiment} | {g.summary}`"]
            for name in ("Keyword baseline (no model)", "distilgpt2, 3-shot prompt, no LoRA", "distilgpt2 + LoRA, merged"):
                p = samples[set_name][name][i]
                sm.append(f"- {name}: `{p.topic} | {p.sentiment} | {p.summary}`")
            sm.append("")
    (out / "samples.md").write_text("\n".join(sm), encoding="utf-8")
    print("wrote results/results.md, results/results.json, results/samples.md")


if __name__ == "__main__":
    main()
