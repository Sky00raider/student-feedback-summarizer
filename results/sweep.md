# LoRA sweep

Same budget for every run: first 400 training examples, 2 epochs, lr 0.002, distilgpt2. Scores: val = 60 template + 32 varied validation comments, free = 48 independent freeform comments.

| Run | Trainable params | Train s | Val loss | Val topic % | Val sent. % | Val ROUGE-L | Free topic % | Free sent. % | Free ROUGE-L | Free words in comment % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r=2 alpha=4 | 147,456 | 166 | 1.522 | 50.0 | 63.0 | 44.2 | 43.8 | 64.6 | 27.3 | 92.3 |
| r=8 alpha=16 | 589,824 | 153 | 1.298 | 58.7 | 62.0 | 46.2 | 62.5 | 64.6 | 29.1 | 92.5 |
| r=32 alpha=64 | 2,359,296 | 1541 | 1.129 | 69.6 | 71.7 | 51.2 | 64.6 | 75.0 | 29.9 | 87.8 |
| r=8 alpha=64 | 589,824 | 173 | 1.150 | 69.6 | 68.5 | 50.2 | 58.3 | 68.8 | 29.2 | 88.3 |
| r=8 alpha=16, template data only | 589,824 | 173 | 1.431 | 58.7 | 73.9 | 47.3 | 52.1 | 60.4 | 23.7 | 79.5 |
