# Results

Adapter: `models/adapter.pt`, r=16, alpha=32, targets c_attn, c_proj, c_fc, 888 training examples, 3 epochs, trained in 529 s on a laptop CPU.

## synthetic test (held-out phrasings + courses)

| System | Topic acc % | Sentiment acc % | Issue-vs-praise acc % | ROUGE-1 | ROUGE-2 | ROUGE-L | Well-formed % | Words found in comment % | Words / summary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Keyword baseline (no model) | 68.3 | 55.8 | 59.2 | 46.2 | 26.2 | 42.6 | 100.0 | 100.0 | 11.98 |
| distilgpt2, 3-shot prompt, no LoRA | 1.7 | 10.8 | 40.0 | 14.9 | 1.7 | 13.5 | 33.3 | 15.3 | 11.51 |
| distilgpt2 + LoRA (unmerged) | 71.7 | 88.3 | 91.7 | 41.9 | 21.3 | 40.6 | 100.0 | 81.7 | 9.95 |
| distilgpt2 + LoRA, merged | 71.7 | 88.3 | 91.7 | 41.9 | 21.3 | 40.6 | 100.0 | 81.7 | 9.95 |
| Reference summaries (upper bound) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 71.5 | 10.50 |

## freeform (48 comments written independently)

| System | Topic acc % | Sentiment acc % | Issue-vs-praise acc % | ROUGE-1 | ROUGE-2 | ROUGE-L | Well-formed % | Words found in comment % | Words / summary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Keyword baseline (no model) | 64.6 | 62.5 | 70.8 | 30.0 | 9.6 | 25.2 | 100.0 | 100.0 | 11.35 |
| distilgpt2, 3-shot prompt, no LoRA | 2.1 | 8.3 | 35.4 | 6.5 | 0.1 | 5.2 | 27.1 | 4.0 | 8.58 |
| distilgpt2 + LoRA (unmerged) | 93.8 | 60.4 | 83.3 | 35.1 | 13.6 | 31.3 | 100.0 | 86.8 | 13.10 |
| distilgpt2 + LoRA, merged | 93.8 | 60.4 | 83.3 | 35.1 | 13.6 | 31.3 | 100.0 | 86.8 | 13.10 |
| Reference summaries (upper bound) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 65.4 | 14.65 |

ROUGE is F1 x 100 against the reference summary. 'Words found in comment' = share of the summary's content words that appear in the original comment (a hallucination proxy); the reference summaries paraphrase, so they score below 100 too. 'Issue-vs-praise' counts negative and mixed as one class: that is all the digest needs to decide whether a comment is an action item.

## Merging (milestone 5)

- 24 LoRA layers merged into the weights (W' = W + alpha/r * B @ A).
- Max |logit difference| merged vs unmerged: 9.2e-05.
- Identical generated text: 120/120 on synthetic test, 48/48 on freeform.
- Latency (batch of 8, greedy, CPU): 837 ms/comment unmerged, 470 ms/comment merged.
- Adapter file 4626 KB (1,179,648 trainable parameters, 1.42% of 83,092,224); merged model = a plain distilgpt2 checkpoint.
