"""Tests for app/core.py, one group per milestone. No network: the model is a tiny random GPT-2 and the
tokenizer works on raw bytes, so nothing is downloaded."""
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from app import core
from app.core import Example


class ByteTokenizer:
    """Stand-in for the GPT-2 tokenizer: one id per byte, 256 = end of text."""
    eos_token_id = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=True):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="ignore")

    def save_pretrained(self, directory):
        pass


def tiny_gpt2(seed=0):
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=257, n_positions=512, n_embd=32, n_layer=2, n_head=2,
                     bos_token_id=256, eos_token_id=256, pad_token_id=256)
    return GPT2LMHeadModel(cfg).eval()


EX = Example("t1", "the pace in week 3 was way too fast", "pacing", "negative", "Found week 3 too fast.")


# ---------------------------------------------------------------- Milestone 1: synthetic dataset
def test_dataset_has_the_requested_sizes_and_covers_every_label():
    d = core.make_dataset(120, 20, 40, seed=1)
    assert [len(d[s]) for s in ("train", "val", "test")] == [120, 20, 40]
    train = d["train"]
    assert {e.topic for e in train} == set(core.TOPICS)
    assert {e.sentiment for e in train} == set(core.SENTIMENTS)
    assert len({e.style for e in train}) >= 6


def test_dataset_is_deterministic_and_has_no_duplicates_across_splits():
    a, b = core.make_dataset(60, 10, 20, seed=3), core.make_dataset(60, 10, 20, seed=3)
    assert [e.comment for e in a["train"]] == [e.comment for e in b["train"]]
    comments = [e.comment.lower() for s in a.values() for e in s]
    assert len(comments) == len(set(comments))


def test_test_split_uses_held_out_phrasings_and_courses():
    d = core.make_dataset(200, 10, 60, seed=0)
    assert all(e.course in core.TEST_COURSES for e in d["test"])
    assert not any(e.course in core.TEST_COURSES for e in d["train"])
    held_out = [core.CLAIMS[t][p][-1][0] for t in core.TOPICS for p in ("neg", "pos")]
    probes = [h[:30].lower() for h in held_out if "{" not in h[:30]]
    train_text = " ".join(e.comment.lower() for e in d["train"])
    assert probes and not any(p in train_text for p in probes)


def test_mixed_examples_join_praise_and_complaint_and_basic_has_none():
    full = core.make_dataset(200, 10, 10, seed=0, variety="full")["train"]
    mixed = [e for e in full if e.sentiment == "mixed"]
    assert mixed and all(" but " in e.summary for e in mixed)
    basic = core.make_dataset(100, 10, 10, seed=0, variety="basic")
    assert {e.sentiment for e in basic["train"]} <= {"positive", "negative"}
    assert {e.style for e in basic["train"]} == {"formal"}


def test_summaries_are_shorter_than_the_comment_and_jsonl_round_trips(tmp_path):
    rows = core.make_dataset(100, 5, 5, seed=2)["train"]
    stats = core.dataset_stats(rows)
    assert stats["summary_words_mean"] < stats["comment_words_mean"]
    core.save_jsonl(rows[:5], tmp_path / "x.jsonl")
    assert core.load_jsonl(tmp_path / "x.jsonl") == rows[:5]


def test_mixing_keeps_varied_validation_examples_out_of_training():
    template = core.make_dataset(40, 2, 2, seed=0)["train"]
    diverse = [Example(f"d{i}", f"varied comment {i}", core.TOPICS[i % 8], "negative", "s") for i in range(32)]
    train, val = core.mix_datasets(template, diverse, n_template=30, n_diverse_val=16)
    assert len(val) == 16 and {e.topic for e in val} == set(core.TOPICS)
    assert len(train) == 30 + 16 and not {e.id for e in val} & {e.id for e in train}


def test_committed_varied_training_sets_do_not_overlap_the_test_set():
    diverse = core.load_diverse(Path("dataset"))
    free = {e.comment for e in core.load_jsonl(Path("dataset/freeform.jsonl"))}
    assert len(diverse) == 320 and not free & {e.comment for e in diverse}
    assert all(e.topic in core.TOPICS and e.sentiment in core.SENTIMENTS for e in diverse)


def test_committed_freeform_set_is_well_formed():
    rows = core.load_jsonl(Path("dataset/freeform.jsonl"))
    assert len(rows) == 48
    assert all(r.topic in core.TOPICS and r.sentiment in core.SENTIMENTS for r in rows)
    assert all(len(r.summary.split()) <= 20 for r in rows)


# ---------------------------------------------------------------- Milestone 2: LoRA injection + training
def test_fresh_lora_changes_nothing_and_only_a_b_are_trainable():
    model = tiny_gpt2()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    before = model(input_ids=ids).logits
    names = core.inject_lora(model, ("c_attn",), r=4, alpha=8)
    assert len(names) == 2
    assert torch.allclose(model(input_ids=ids).logits, before)          # B = 0 at start
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith((".A", ".B")) for n in trainable)
    lora, total = core.count_parameters(model)
    assert lora == 2 * 4 * (32 + 96) and lora < total


def test_lora_wraps_linear_layers_too():
    wrapped = core.LoRALinear(torch.nn.Linear(6, 5), r=2, alpha=4)
    assert wrapped.delta_weight().shape == (5, 6)
    with pytest.raises(TypeError):
        core.LoRALinear(torch.nn.ReLU())


def test_labels_mask_the_prompt_and_end_with_eos():
    tok = ByteTokenizer()
    ids, labels = core.encode_example(tok, EX)
    n_prompt = len(core.prompt_ids(tok, EX.comment))
    assert labels[:n_prompt] == [-100] * n_prompt
    assert labels[n_prompt:] == ids[n_prompt:] and ids[-1] == tok.eos_token_id
    assert tok.decode(ids[n_prompt:]) == " pacing | negative | Found week 3 too fast."


def test_long_comments_are_cut_but_keep_the_summary_cue():
    tok = ByteTokenizer()
    ids = core.prompt_ids(tok, "word " * 500, max_prompt_tokens=60)
    assert len(ids) == 60 and tok.decode(ids).endswith("\nSummary:")


def test_training_lowers_the_loss_and_the_adapter_round_trips(tmp_path):
    tok = ByteTokenizer()
    train = [EX, Example("t2", "loved the slides", "materials", "positive", "Liked the slides.")] * 4
    cfg = core.TrainConfig(r=4, alpha=8, lr=5e-3, epochs=6, batch_size=4)
    model = tiny_gpt2()
    hist = core.train_lora(model, tok, train, cfg, val=train[:2], log=None)
    assert hist["val_loss"][-1] < hist["val_loss"][0]
    size = core.save_adapter(model, cfg, tmp_path / "a.pt")
    assert 0 < size < 200_000
    fresh = tiny_gpt2()
    loaded_cfg = core.apply_adapter(fresh, core.read_adapter(tmp_path / "a.pt"))
    ids = torch.tensor([core.prompt_ids(tok, EX.comment)])
    assert loaded_cfg.r == 4
    assert torch.allclose(fresh(input_ids=ids).logits, model(input_ids=ids).logits, atol=1e-6)


def test_adapter_with_wrong_shapes_is_refused(tmp_path):
    model = tiny_gpt2()
    core.inject_lora(model, ("c_attn",), r=4)
    core.save_adapter(model, core.TrainConfig(r=4), tmp_path / "a.pt")
    saved = core.read_adapter(tmp_path / "a.pt")
    saved["config"]["r"] = 2
    with pytest.raises(ValueError):
        core.apply_adapter(tiny_gpt2(), saved)


# ---------------------------------------------------------------- Milestone 3: summarization pipeline (UI backend)
def test_read_feedback_finds_the_comment_column_in_csv():
    data = "student,rating,comment\nA,4,Great lectures\nB,2,\"Too fast, honestly\"\n".encode()
    assert core.read_feedback("f.csv", data) == ["Great lectures", "Too fast, honestly"]
    no_header = b"1,the pace was too fast for me\n2,slides were great\n"
    assert core.read_feedback("f.csv", no_header) == ["the pace was too fast for me", "slides were great"]


def test_read_feedback_txt_lines_paragraphs_and_json():
    assert core.read_feedback("f.txt", "one\n\ntwo\n") == ["one", "two"]
    assert core.read_feedback("f.txt", "first line\nsecond\n\nthird\n") == ["first line second", "third"]
    assert core.read_feedback("f.txt", "a\nb\n") == ["a", "b"]
    assert core.read_feedback("f.jsonl", '{"feedback": "x"}\n{"comment": "y"}\n') == ["x", "y"]
    assert core.read_feedback("f.json", '["p", "q"]') == ["p", "q"]


def test_redact_removes_personal_details():
    text, n = core.redact("My name is Priya Raman, email priya.r@uni.edu, ID S1234567, call +91 98765 43210.")
    assert "Priya" not in text and "@" not in text and "1234567" not in text and "98765" not in text
    assert n == 4
    assert core.redact("The pace in week 3 was fast.") == ("The pace in week 3 was fast.", 0)


def test_parse_output_accepts_the_format_and_flags_garbage():
    r = core.parse_output(" pacing | negative | Found week 3 too fast.", EX.comment)
    assert (r.topic, r.sentiment, r.summary, r.valid) == ("pacing", "negative", "Found week 3 too fast.", True)
    bad = core.parse_output(" the the the", EX.comment)
    assert not bad.valid and bad.topic == "unknown"
    assert not core.parse_output(" cooking | negative | x").valid


def test_repeated_halves_of_a_summary_are_collapsed():
    assert core.clean_summary("Says lab computers lacked Python but says lab computers lacked Python.") ==         "Says lab computers lacked Python."
    assert core.clean_summary("Praises the slides but says the midterm was unfair.") ==         "Praises the slides but says the midterm was unfair."
    assert core.clean_summary("Found the the slides slides useful.") == "Found the slides useful."
    r = core.parse_output(" technology | mixed | Says X broke but says X broke.", "X broke")
    assert r.summary == "Says X broke." and r.valid


def test_unsupported_words_catches_invented_content():
    assert core.unsupported_words("Found week 3 too fast.", EX.comment) == []
    assert core.unsupported_words("Hates the chemistry textbook.", EX.comment) == ["hates", "chemistry", "textbook"]


def test_summarizer_runs_end_to_end_on_a_tiny_model():
    s = core.Summarizer(tiny_gpt2(), ByteTokenizer(), max_new_tokens=5, batch_size=2)
    out = s.summarize(["short", "a somewhat longer comment", "mail me at a@b.co"])
    assert len(out) == 3 and out[2].redactions == 1 and "a@b.co" not in out[2].comment
    single = core.Summarizer(s.model, s.tok, max_new_tokens=5, batch_size=1).raw(["short", "a somewhat longer comment"])
    assert single == s.raw(["short", "a somewhat longer comment"])        # left padding does not change outputs


def test_streamlit_page_renders_upload_and_ratings_tabs(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "app.json"))
    at = AppTest.from_file(str(Path("app/main.py").resolve()), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Student Feedback Summarizer"
    assert len(at.tabs) == 2 and at.text_area and at.button


def _fs(topic, sentiment, summary="s", valid=True):
    return core.FeedbackSummary("c", topic, sentiment, summary, valid=valid)


def test_digest_ranks_action_items_by_how_many_students_raised_them():
    results = [_fs("pacing", "negative", "too fast")] * 3 + [_fs("support", "mixed", "slow emails")] \
        + [_fs("materials", "positive", "great slides")] * 2 + [_fs("unknown", "unknown", valid=False)]
    d = core.build_digest(results)
    assert [a["topic"] for a in d["action_items"]] == ["pacing", "support"]
    assert d["action_items"][0]["count"] == 3
    assert d["strengths"][0]["topic"] == "materials"
    assert d["sentiment"] == {"positive": 2, "negative": 3, "mixed": 1} and len(d["needs_review"]) == 1
    md = core.digest_markdown(d)
    assert "1. **pacing** (3 students)" in md and "Read these yourself (1)" in md


# ---------------------------------------------------------------- Milestone 4: evaluation
def test_rouge_scores_known_cases():
    assert core.rouge_n("the cat sat", "the cat sat", 1) == 1.0
    assert core.rouge_n("the cat sat", "dog", 1) == 0.0
    assert core.rouge_n("a b c d", "a b x y", 2) == pytest.approx(1 / 3)
    assert core.rouge_l("a b c d", "a x c d") == pytest.approx(0.75)


def test_evaluate_scores_perfect_and_wrong_predictions():
    gold = [EX, Example("t2", "loved the slides", "materials", "positive", "Liked the slides.")]
    perfect = core.evaluate(core.gold_as_predictions(gold), gold)
    assert perfect["topic_acc"] == perfect["sentiment_acc"] == perfect["rouge1"] == perfect["valid_format"] == 1.0
    wrong = core.evaluate([_fs("support", "mixed", "nothing related"), _fs("x", "y", "q", valid=False)], gold)
    assert wrong["topic_acc"] == 0 and wrong["valid_format"] == 0.5 and wrong["rouge1"] == 0
    assert wrong["issue_acc"] == 1.0          # "mixed" for a negative comment still counts as an issue
    assert core.confusion([_fs("support", "negative")], [EX])["pacing"] == {"support": 1}


def test_keyword_baseline_gets_obvious_cases():
    r = core.keyword_baseline("The pace in week 3 was way too fast.")
    assert (r.topic, r.sentiment) == ("pacing", "negative")
    assert core.keyword_baseline("The slides were great, but the midterm was confusing.").sentiment == "mixed"
    assert core.keyword_baseline("The TA was really helpful.").topic == "support"
    assert len(core.keyword_baseline("word " * 40).summary.split()) == 20
