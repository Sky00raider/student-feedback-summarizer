# Student Feedback Summarizer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/kgqfvkcj6d/badge.svg)](https://ask.thili.ai/b/kgqfvkcj6d)

This project, "Student Feedback Summarizer," provides course instructors with a powerful tool to streamline curriculum improvement. By fine-tuning a language model with LoRA on course feedback, it generates concise summaries of student comments, demonstrating the practical application of LoRA for domain-specific text summarization.

## What you are building

**Goal.** You will have built a tool that fine-tunes a language model using LoRA to summarize student feedback, helping course instructors rapidly improve curriculum.

**Done when.** The project is complete when the fine-tuned LoRA adapter can generate concise and relevant summaries of student comments, proven by successful training, adapter merging, and evaluation against original feedback.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a small dataset of synthetic student feedback and summary pairs.**  
      Hint: Focus on creating diverse examples for effective training.
- [ ] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Use the `lfllm-lflora` technique to inject and train the adapter.
- [ ] **3. Develop a Streamlit UI to upload feedback and trigger summarization.**  
      Hint: Use Streamlit components for file upload and text display.
- [ ] **4. Evaluate the quality of generated summaries against original feedback.**  
      Hint: Compare generated summaries with your ground truth summaries.
- [ ] **5. Implement adapter merging for inference.**  
      Hint: Apply the `lfllm-lflora` merging technique to prepare the model for deployment.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Experiment with different LoRA hyper-parameters (e.g., rank, alpha) to observe their impact on summary quality.
- Expand the synthetic dataset with more varied feedback styles and topics to improve the model's generalization.
- Implement a user feedback mechanism within the Streamlit UI for instructors to rate the quality of generated summaries.
