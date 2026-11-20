"""Student Feedback Summarizer: the code, one section per milestone.

Every comment becomes one structured line an instructor can act on:

    "honestly the lectures r great but week 4 went way too fast"
      ->  pacing | mixed | Found the lectures clear but found week 4 too fast to keep up with.

A small GPT-2 (distilgpt2) learns that format through a LoRA adapter written by hand below. The per-comment
lines are then aggregated into a course digest: action items ranked by how many students raised them.
"""
from __future__ import annotations

import csv
import io
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

TOPICS = ("pacing", "workload", "lectures", "assignments", "assessment", "materials", "support", "technology")
SENTIMENTS = ("positive", "negative", "mixed")
BASE_MODEL = "distilbert/distilgpt2"
DEFAULT_ADAPTER = Path("models/adapter.pt")
DATASET_DIR = Path("dataset")


# =====================================================================================================
# --- Milestone 1: Prepare a small dataset of synthetic student feedback and summary pairs.
# =====================================================================================================
# Each topic has negative and positive "claims": (what a student writes, how a summary says it).
# The comment side is a clause ("the pace felt just right"); the summary side is a verb phrase about the
# student ("felt the pace was right"), so a mixed comment's summary is "<praise> but <complaint>."
# The LAST claim of every list is held out: it only ever appears in the test split, so the test measures
# phrasings the model never saw, not memorised templates.

CLAIMS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "pacing": {
        "neg": [("the pace in week {week} was way too fast", "found week {week} too fast to keep up with"),
                ("we rushed through {sub} and I never caught up", "felt {sub} was rushed"),
                ("{sub} was crammed into a single lecture", "wants more than one lecture on {sub}"),
                ("the second half of the term moved much faster than the first", "found the second half of the term too fast"),
                ("the course dragged at the start and then sprinted at the end", "found the pacing uneven, slow early and rushed late")],
        "pos": [("the pace felt just right", "felt the pace was right"),
                ("we spent enough time on {sub} before moving on", "appreciated the time spent on {sub}"),
                ("the course never felt rushed, even in week {week}", "never felt the course was rushed"),
                ("each week built steadily on the last", "liked how each week built on the last"),
                ("the slower review weeks before exams really helped", "valued the slower review weeks before exams")],
    },
    "workload": {
        "neg": [("the workload was far too heavy for a {credits}-credit course", "finds the workload too heavy for a {credits}-credit course"),
                ("I spent over {hours} hours a week on this class", "spends over {hours} hours a week on the course"),
                ("having a quiz, a lab and a problem set due every week was exhausting", "finds weekly quizzes, labs and problem sets exhausting together"),
                ("the reading load for {sub} was unrealistic", "finds the {sub} reading load unrealistic"),
                ("deadlines from three deliverables landed in the same week", "wants deliverables spread out instead of clustered in one week")],
        "pos": [("the workload was manageable alongside my other classes", "finds the workload manageable"),
                ("about {hours} hours a week felt fair for what we learned", "feels {hours} hours a week is fair"),
                ("deadlines were spread out sensibly", "appreciated sensibly spread deadlines"),
                ("the weekly tasks were small enough to finish on time", "liked that weekly tasks were small and doable"),
                ("dropping the lowest homework score took a lot of pressure off", "appreciated dropping the lowest homework score")],
    },
    "lectures": {
        "neg": [("the lectures on {sub} were hard to follow", "found the {sub} lectures hard to follow"),
                ("the professor read straight off the slides", "dislikes lectures read straight from the slides"),
                ("lectures often ran over time and skipped the examples", "notes lectures ran over time and skipped examples"),
                ("there were no worked examples in lecture, only theory", "wants worked examples in lectures, not only theory"),
                ("the lecturer talked too quietly to hear from the back", "could not hear the lecturer from the back of the room")],
        "pos": [("the lectures on {sub} were really clear", "found the {sub} lectures clear"),
                ("the professor explained {sub} with great real-world examples", "liked the real-world examples for {sub}"),
                ("lectures were engaging and well organised", "found lectures engaging and well organised"),
                ("the live demos in lecture made {sub} click", "says live demos in lecture made {sub} click"),
                ("the short recap at the start of each lecture helped a lot", "values the recap at the start of each lecture")],
    },
    "assignments": {
        "neg": [("the instructions for assignment {n} were confusing", "found the assignment {n} instructions confusing"),
                ("the homework had little to do with what we covered in class", "says homework did not match the class content"),
                ("the {sub} project was far bigger than the time we were given", "felt the {sub} project was too big for the time given"),
                ("assignment {n} had errors in the starter files", "reports errors in the assignment {n} starter files"),
                ("group projects left a few people doing all the work", "says group projects left a few people doing all the work")],
        "pos": [("the assignments were practical and relevant", "found the assignments practical and relevant"),
                ("assignment {n} on {sub} was the best part of the course", "rates assignment {n} on {sub} as the best part"),
                ("the project let us apply {sub} to a real problem", "liked applying {sub} to a real problem in the project"),
                ("each assignment came with clear instructions and examples", "praises clear assignment instructions and examples"),
                ("being allowed to pick our own project topic was motivating", "found choosing their own project topic motivating")],
    },
    "assessment": {
        "neg": [("the midterm covered material we never saw in class", "says the midterm covered material not taught in class"),
                ("grading was inconsistent between TAs", "finds grading inconsistent between TAs"),
                ("we got our grades back with no feedback at all", "wants feedback with grades, not just scores"),
                ("the final exam was far too long for two hours", "found the final exam too long for the time"),
                ("the rubric for the essay was only shared after grading", "wants the essay rubric shared before grading")],
        "pos": [("the exams were fair and matched the practice problems", "found exams fair and matched to practice problems"),
                ("grading was quick and the comments were useful", "praises quick grading with useful comments"),
                ("the quizzes on {sub} were a good check of understanding", "found the {sub} quizzes a good check of understanding"),
                ("the rubrics made it clear what was expected", "says rubrics made expectations clear"),
                ("being able to retake one quiz reduced exam stress", "appreciated being able to retake a quiz")],
    },
    "materials": {
        "neg": [("the slides for {sub} were missing half the content", "found the {sub} slides incomplete"),
                ("the textbook was expensive and we barely used it", "says the expensive textbook was barely used"),
                ("the lecture notes were posted too late to prepare", "wants lecture notes posted earlier"),
                ("several of the recorded videos had the wrong week's content", "reports recorded videos with the wrong week's content"),
                ("the readings were outdated and full of broken links", "found readings outdated with broken links")],
        "pos": [("the slides were detailed enough to study from", "found the slides detailed enough to study from"),
                ("the lecture notes on {sub} were excellent", "praises the {sub} lecture notes"),
                ("the recorded videos made revision easy", "found the recorded videos useful for revision"),
                ("the extra readings on {sub} were interesting", "enjoyed the extra readings on {sub}"),
                ("the cheat sheet summarising each week was a lifesaver", "values the weekly summary cheat sheet")],
    },
    "support": {
        "neg": [("office hours were always full and I never got help", "could not get help at crowded office hours"),
                ("emails to the instructor went unanswered for weeks", "says emails to the instructor went unanswered"),
                ("questions on the forum sat for days without a reply", "says forum questions went days without replies"),
                ("the TAs did not seem to understand {sub} themselves", "doubts the TAs' grasp of {sub}"),
                ("there was no help available in the week before the exam", "wants help available in the week before the exam")],
        "pos": [("the TAs were patient and helpful in office hours", "praises patient, helpful TAs in office hours"),
                ("the instructor answered emails within a day", "appreciates email replies within a day"),
                ("the forum was active and questions got answered fast", "found the forum active with fast answers"),
                ("the extra help session on {sub} saved me", "says the extra help session on {sub} was valuable"),
                ("the professor clearly cared about whether we understood", "felt the professor cared about student understanding")],
    },
    "technology": {
        "neg": [("{tool} kept crashing when I tried to submit", "reports {tool} crashing on submission"),
                ("the lab computers did not have the software we needed", "says lab computers lacked the needed software"),
                ("the audio on the lecture recordings was terrible", "found the lecture recording audio poor"),
                ("setting up {tool} took most of the first two weeks", "says setting up {tool} took two weeks"),
                ("the online quizzes timed out and lost my answers", "reports online quizzes timing out and losing answers")],
        "pos": [("{tool} made submitting work easy", "found {tool} made submission easy"),
                ("the online platform was well organised", "found the online platform well organised"),
                ("the lab machines were set up and ready every week", "liked that lab machines were ready every week"),
                ("the autograder on {tool} gave instant feedback", "values the instant autograder feedback on {tool}"),
                ("lecture recordings were good quality and posted quickly", "liked good-quality recordings posted quickly")],
    },
}

# Course -> subtopics that fill {sub}. The last two courses only appear in the test split.
COURSES = {
    "Intro to Programming": ["loops", "recursion", "functions", "file handling", "classes"],
    "Statistics 101": ["hypothesis testing", "regression", "probability", "confidence intervals"],
    "Organic Chemistry": ["reaction mechanisms", "stereochemistry", "spectroscopy", "synthesis"],
    "World History": ["the Cold War", "the Industrial Revolution", "colonialism", "the French Revolution"],
    "Linear Algebra": ["eigenvalues", "matrix decompositions", "vector spaces", "determinants"],
    "Machine Learning": ["gradient descent", "neural networks", "decision trees", "regularisation"],
    "Microeconomics": ["elasticity", "game theory", "market structures", "consumer choice"],
    "Data Structures": ["linked lists", "hash tables", "graphs", "heaps"],
}
TEST_COURSES = ("Microeconomics", "Data Structures")
TOOLS = ["Moodle", "Canvas", "Gradescope", "Jupyter", "the course website", "Blackboard"]

SINGLE_TEMPLATES = {          # style -> templates for a one-claim comment ({c} clause, {C} capitalised)
    "formal": ["I felt that {c}.", "In my view, {c}.", "Overall, {c}.", "{C}.", "I would like to note that {c}."],
    "casual": ["honestly {c}", "ngl {c}", "{c} tbh", "So yeah, {c}.", "{C} lol"],
    "emphatic": ["{C}. {X}", "{C}!! {X}", "Seriously, {c}. {X}"],
    "terse": ["{c}", "{C}."],
    "rambling": ["{F} {C}. {G}", "{F} Anyway, {c}. {G}", "{F} {C}, if I'm being honest. {G}"],
}
EXTRA = {"neg": ["Really frustrating.", "Please fix this.", "It made the course much harder.", "Not great."],
         "pos": ["Thank you!", "Keep it up.", "Great job.", "Loved it."]}
MIXED_TEMPLATES = ["{P}, but {n}.", "While {p}, {n}.", "{P}. However, {n}.", "{P}. On the other hand, {n}.",
                   "{N}, although {p}.", "{N}. To be fair, {p}.", "loved that {p} but {n}", "{P}; sadly {n}."]
FILLER_PRE = ["I took this course as an elective this term.", "I'm a second-year student.",
              "This was my first course in the subject.", "Some thoughts on the course.", "Overall it was an okay term."]
FILLER_POST = ["Otherwise the course was fine.", "That's all from me.", "Hope this helps next year.", "Thanks for reading."]
STYLES = {"basic": ["formal"], "full": ["formal", "casual", "emphatic", "terse", "rambling", "lowercase", "typos"]}


@dataclass
class Example:
    id: str
    comment: str
    topic: str
    sentiment: str
    summary: str
    course: str = ""
    style: str = ""


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _fill(text: str, slots: dict) -> str:
    return text.format(**slots)


def _typos(text: str, rng: random.Random, rate: float = 0.08) -> str:
    """Swap two neighbouring letters in roughly `rate` of the longer words."""
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 4 and w.isalpha() and rng.random() < rate:
            j = rng.randrange(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def _claims(topic: str, polarity: str, split: str) -> list[tuple[str, str]]:
    pool = CLAIMS[topic][polarity]
    return pool[-1:] if split == "test" else pool[:-1]


def make_example(rng: random.Random, split: str = "train", variety: str = "full", idx: int = 0) -> Example:
    """One synthetic (comment, topic, sentiment, summary) example."""
    courses = list(TEST_COURSES) if split == "test" else [c for c in COURSES if c not in TEST_COURSES]
    course = rng.choice(courses)
    slots = {"sub": rng.choice(COURSES[course]), "week": rng.randint(2, 11), "n": rng.randint(1, 6),
             "hours": rng.choice([10, 12, 15, 18, 20, 25]), "credits": rng.choice([2, 3]), "tool": rng.choice(TOOLS)}
    topic = rng.choice(TOPICS)
    weights = [0.3, 0.4, 0.3] if variety == "full" else [0.45, 0.55, 0.0]
    sentiment = rng.choices(SENTIMENTS, weights)[0]
    style = rng.choice(STYLES[variety])

    if sentiment == "mixed":            # praise for anything, complaint about `topic` (the actionable part)
        p_topic = rng.choice([t for t in TOPICS if t != topic])
        pc, ps = (_fill(x, slots) for x in rng.choice(_claims(p_topic, "pos", split)))
        nc, ns = (_fill(x, slots) for x in rng.choice(_claims(topic, "neg", split)))
        comment = rng.choice(MIXED_TEMPLATES).format(p=pc, P=_cap(pc), n=nc, N=_cap(nc))
        summary = f"{_cap(ps)} but {ns}."
        if style == "rambling":
            comment = f"{rng.choice(FILLER_PRE)} {comment} {rng.choice(FILLER_POST)}"
    else:
        pol = "pos" if sentiment == "positive" else "neg"
        c, s = (_fill(x, slots) for x in rng.choice(_claims(topic, pol, split)))
        tstyle = style if style in SINGLE_TEMPLATES else rng.choice(["formal", "casual", "terse"])
        comment = rng.choice(SINGLE_TEMPLATES[tstyle]).format(
            c=c, C=_cap(c), X=rng.choice(EXTRA[pol]), F=rng.choice(FILLER_PRE), G=rng.choice(FILLER_POST))
        summary = _cap(s) + "."
    if style == "lowercase":
        comment = re.sub(r"[.,;!]", "", comment.lower())
    elif style == "typos":
        comment = _typos(comment, rng)
    return Example(f"{split}-{idx:04d}", comment, topic, sentiment, summary, course, style)


def make_dataset(n_train: int = 480, n_val: int = 60, n_test: int = 120, seed: int = 0,
                 variety: str = "full") -> dict[str, list[Example]]:
    """Train / val / test splits with no duplicate comments within or across splits.

    variety="basic" is the narrow first version (formal wording, no mixed comments); "full" adds casual,
    terse, emphatic, rambling, lowercase and typo styles plus mixed praise-and-complaint comments.
    The test split is always "full" so both versions are scored on the same kind of text.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    out: dict[str, list[Example]] = {}
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        rows: list[Example] = []
        tries = 0
        while len(rows) < n:
            tries += 1
            if tries > n * 200:
                raise RuntimeError(f"could not make {n} unique {split} examples")
            ex = make_example(rng, split, variety if split != "test" else "full", len(rows))
            key = ex.comment.lower()
            if key not in seen:
                seen.add(key)
                rows.append(ex)
        out[split] = rows
    return out


def save_jsonl(rows: list[Example], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[Example]:
    fields_ = Example.__dataclass_fields__
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            rows.append(Example(**{k: v for k, v in d.items() if k in fields_}))
    return rows


def mix_datasets(template: list[Example], diverse: list[Example], n_template: int = 600, n_diverse_val: int = 32,
                 seed: int = 0) -> tuple[list[Example], list[Example]]:
    """Stretch: template examples + hand-written varied ones (dataset/diverse_*.jsonl).
    Returns (train, diverse_val): a few varied examples are kept back for validation, stratified by topic."""
    rng = random.Random(seed)
    by_topic: dict[str, list[Example]] = defaultdict(list)
    for e in diverse:
        by_topic[e.topic].append(e)
    val = []
    for t in TOPICS:
        rng.shuffle(by_topic[t])
        val += by_topic[t][: n_diverse_val // len(TOPICS)]
    held = {e.id for e in val}
    train = template[:n_template] + [e for e in diverse if e.id not in held]
    rng.shuffle(train)
    return train, val


def load_diverse(directory: Path = DATASET_DIR) -> list[Example]:
    return [e for f in sorted(Path(directory).glob("diverse_*.jsonl")) for e in load_jsonl(f)]


def dataset_stats(rows: list[Example]) -> dict:
    words = [len(r.comment.split()) for r in rows]
    return {"n": len(rows), "topics": dict(Counter(r.topic for r in rows)),
            "sentiments": dict(Counter(r.sentiment for r in rows)), "styles": dict(Counter(r.style for r in rows)),
            "comment_words_mean": round(sum(words) / max(len(words), 1), 1),
            "summary_words_mean": round(sum(len(r.summary.split()) for r in rows) / max(len(rows), 1), 1)}


# =====================================================================================================
# --- Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
# =====================================================================================================
# LoRA by hand, no peft:  y = base(x) + (alpha / r) * (x A^T) B^T, with A (r x in) small random and
# B (out x r) zero, so the wrapped model starts out computing exactly what the pretrained one did.
# GPT-2 uses a `Conv1D` whose weight is stored (in, out); forward never touches it, merging transposes.
# Training: the loss is masked to the summary tokens only (-100 on the prompt), so the model learns to
# write summaries, not to imitate student comments.
import torch                                                                    # noqa: E402
from torch import nn                                                            # noqa: E402

PROMPT_PREFIX, PROMPT_SUFFIX = "Feedback: ", "\nSummary:"


def is_conv1d(layer: nn.Module) -> bool:
    return type(layer).__name__ == "Conv1D"


def layer_shape(layer: nn.Module) -> tuple[int, int]:
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features
    if is_conv1d(layer):
        n_in, n_out = layer.weight.shape
        return n_in, n_out
    raise TypeError(f"LoRA does not know how to wrap {type(layer).__name__}")


class LoRALinear(nn.Module):
    """A frozen base layer plus a trainable low-rank update B @ A."""

    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.in_features, self.out_features = layer_shape(base)
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        w = base.weight
        self.A = nn.Parameter(torch.randn(r, self.in_features, dtype=w.dtype, device=w.device) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.out_features, r, dtype=w.dtype, device=w.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    def delta_weight(self) -> torch.Tensor:
        return self.scale * (self.B @ self.A)               # (out, in)

    @torch.no_grad()
    def merged(self) -> nn.Module:
        delta = self.delta_weight()
        self.base.weight += delta.T if is_conv1d(self.base) else delta
        return self.base

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, scale={self.scale:g}"


def _replace(model: nn.Module, name: str, new: nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, new)


def inject_lora(model: nn.Module, targets=("c_attn",), r: int = 8, alpha: float = 16.0) -> list[str]:
    """Freeze everything, then wrap every layer whose name ends in one of `targets`."""
    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, m in model.named_modules()
             if any(n == t or n.endswith("." + t) for t in targets) and not isinstance(m, LoRALinear)]
    if not names:
        raise ValueError(f"no layer called {targets} in the model")
    for name in names:
        _replace(model, name, LoRALinear(model.get_submodule(name), r, alpha))
    return names


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.endswith((".A", ".B"))}


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable LoRA parameters, all parameters), tied tensors counted once."""
    seen, lora, total = set(), 0, 0
    for n, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        lora += p.numel() if n.endswith((".A", ".B")) else 0
    return lora, total


@dataclass
class TrainConfig:
    r: int = 8
    alpha: float = 16.0
    targets: tuple = ("c_attn",)
    lr: float = 1e-3
    epochs: int = 3
    batch_size: int = 8
    max_prompt_tokens: int = 160
    warmup: float = 0.06
    seed: int = 0
    base: str = BASE_MODEL


def target_text(ex: Example) -> str:
    return f" {ex.topic} | {ex.sentiment} | {ex.summary}"


def prompt_ids(tok, comment: str, max_prompt_tokens: int = 160) -> list[int]:
    """'Feedback: <comment>\\nSummary:' as ids; a long comment is cut, the suffix always stays."""
    body = tok.encode(" ".join(comment.split()))
    pre, suf = tok.encode(PROMPT_PREFIX), tok.encode(PROMPT_SUFFIX)
    body = body[: max(1, max_prompt_tokens - len(pre) - len(suf))]
    return pre + body + suf


def encode_example(tok, ex: Example, max_prompt_tokens: int = 160) -> tuple[list[int], list[int]]:
    """(input ids, labels): labels are -100 on the prompt, the target ids + EOS after it."""
    p = prompt_ids(tok, ex.comment, max_prompt_tokens)
    t = tok.encode(target_text(ex)) + [tok.eos_token_id]
    return p + t, [-100] * len(p) + t


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    n = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)], labels[i, :len(y)], mask[i, :len(x)] = torch.tensor(x), torch.tensor(y), 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


@torch.no_grad()
def eval_loss(model, tok, examples: list[Example], cfg: TrainConfig) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(examples), cfg.batch_size):
        batch = collate([encode_example(tok, e, cfg.max_prompt_tokens) for e in examples[i:i + cfg.batch_size]],
                        tok.eos_token_id)
        n_tok = int((batch["labels"][:, 1:] != -100).sum())
        total += float(model(**batch).loss) * n_tok
        count += n_tok
    return total / max(count, 1)


def train_lora(model, tok, train: list[Example], cfg: TrainConfig, val: list[Example] | None = None,
               log=print) -> dict:
    """Inject LoRA (if not there yet) and train A/B with AdamW + warmup/linear decay. Returns a history."""
    torch.manual_seed(cfg.seed)
    if not any(isinstance(m, LoRALinear) for m in model.modules()):
        inject_lora(model, cfg.targets, cfg.r, cfg.alpha)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    encoded = [encode_example(tok, e, cfg.max_prompt_tokens) for e in train]
    steps = cfg.epochs * math.ceil(len(encoded) / cfg.batch_size)
    warm = max(1, int(cfg.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    rng = random.Random(cfg.seed)
    history = {"step_loss": [], "val_loss": [], "seconds": 0.0, "steps": steps}
    t0, step = time.time(), 0
    for epoch in range(cfg.epochs):
        model.train()
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for i in range(0, len(order), cfg.batch_size):
            batch = collate([encoded[j] for j in order[i:i + cfg.batch_size]], tok.eos_token_id)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            history["step_loss"].append(round(loss.item(), 4))
            if log and (step % 10 == 0 or step == steps):
                log(f"epoch {epoch + 1} step {step}/{steps} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
        if val:
            history["val_loss"].append(round(eval_loss(model, tok, val, cfg), 4))
            if log:
                log(f"epoch {epoch + 1} val loss {history['val_loss'][-1]:.3f}")
    history["seconds"] = round(time.time() - t0, 1)
    model.eval()
    return history


def load_base(name: str = BASE_MODEL):
    """Pretrained GPT-2 + tokenizer (downloads ~330 MB the first time)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    return model.eval(), tok


def save_adapter(model, cfg: TrainConfig, path: Path = DEFAULT_ADAPTER, meta: dict | None = None) -> int:
    """Only the LoRA tensors + the settings to rebuild the rest. Returns the file size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "config": asdict(cfg), "lora": lora_state(model), "meta": meta or {}}, path)
    return path.stat().st_size


@torch.no_grad()
def apply_adapter(model, saved: dict) -> TrainConfig:
    """Inject LoRA layers into a fresh base model and copy the saved A/B in, checking names and shapes."""
    cfg = TrainConfig(**{**saved["config"], "targets": tuple(saved["config"]["targets"])})
    inject_lora(model, cfg.targets, cfg.r, cfg.alpha)
    params = dict(model.named_parameters())
    for name, t in saved["lora"].items():
        if name not in params or params[name].shape != t.shape:
            raise ValueError(f"adapter tensor {name} {tuple(t.shape)} does not fit this model")
        params[name].copy_(t)
    model.eval()
    return cfg


def read_adapter(path: Path = DEFAULT_ADAPTER) -> dict:
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError(f"unknown adapter format in {path}")
    return saved


def load_adapter(path: Path = DEFAULT_ADAPTER, merge: bool = True):
    """Fresh pretrained base + saved adapter (merged by default). Returns (model, tokenizer, cfg)."""
    saved = read_adapter(path)
    model, tok = load_base(saved["config"]["base"])
    cfg = apply_adapter(model, saved)
    if merge:
        merge_lora(model)
    return model, tok, cfg


# =====================================================================================================
# --- Milestone 3: Develop a Streamlit UI to upload feedback and trigger summarization.
# =====================================================================================================
# The UI lives in app/main.py; everything it calls is here so it can be tested without a browser:
# reading uploads, redacting personal details, batched generation, parsing, and the course digest.

COMMENT_COLUMNS = ("comment", "comments", "feedback", "response", "text", "answer", "review")
_OUT_RE = re.compile(r"^\s*([a-z]+)\s*\|\s*([a-z]+)\s*\|\s*(.+?)\s*$", re.S)


@dataclass
class FeedbackSummary:
    comment: str
    topic: str
    sentiment: str
    summary: str
    raw: str = ""
    valid: bool = True
    unsupported: list = field(default_factory=list)
    redactions: int = 0


def read_feedback(name: str, data: bytes | str) -> list[str]:
    """Comments from an uploaded .csv (a comment-like column, else the wordiest), .json/.jsonl, or .txt
    (one per line, or paragraphs when blank lines separate them)."""
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
    ext = Path(name).suffix.lower()
    if ext == ".csv":
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            return []
        header = [h.strip().lower() for h in rows[0]]
        col = next((header.index(c) for c in COMMENT_COLUMNS if c in header), None)
        body = rows[1:]
        if col is None:                     # no known header: pick the column with the most words
            if not any(h.isalpha() for h in header):
                body = rows
            width = max(len(r) for r in rows)
            col = max(range(width), key=lambda j: sum(len(r[j].split()) for r in body if j < len(r)))
        comments = [r[col] for r in body if col < len(r)]
    elif ext in (".json", ".jsonl"):
        items = ([json.loads(l) for l in text.splitlines() if l.strip()] if ext == ".jsonl"
                 else json.loads(text))
        comments = [i if isinstance(i, str) else next((i[k] for k in COMMENT_COLUMNS if k in i), "") for i in items]
    else:
        parts = re.split(r"\n\s*\n", text) if re.search(r"\n\s*\n", text.strip()) else text.splitlines()
        comments = [" ".join(p.split()) for p in parts]
    return [c.strip() for c in comments if c and c.strip()]


_PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"(?<!\w)(?:\+?\d[\d\s-]{8,}\d)(?!\w)"), "[phone]"),
    (re.compile(r"\b[A-Z]{1,4}\d{5,}\b|\b\d{7,}\b"), "[id]"),
    (re.compile(r"\b([Mm]y name is|I am|I'm|[Tt]his is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"), r"\1 [name]"),
]


def redact(text: str) -> tuple[str, int]:
    """Remove emails, phone numbers, student IDs and self-introductions. Returns (text, how many)."""
    n = 0
    for pattern, repl in _PII:
        text, k = pattern.subn(repl, text)
        n += k
    return text, n


_STOP = set("""a an the and or but of to in on for with at by from is was were are be been it its this that
these those i me my we our you your they their he she his her them as so too very really just not no if then
than about all any some more most much many there here also only even still over into out up down off do did
does done have has had having will would can could should may might must am being one two own same such week
course class while although however though wasn't didn't don't""".split())
_SUMMARY_VERBS = set("""found finds felt feels says said wants wanted liked likes praises praised appreciated
appreciates values valued enjoyed enjoys notes noted reports reported dislikes disliked rates doubts could
never thinks thought considers student students instructor""".split())


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def unsupported_words(summary: str, comment: str) -> list[str]:
    """Content words of the summary that never appear in the comment (a cheap hallucination check)."""
    source = {_stem(w) for w in _words(comment)}
    return [w for w in _words(summary)
            if w not in _STOP and w not in _SUMMARY_VERBS and _stem(w) not in source and not w.isdigit()]


_LEAD_VERB = re.compile(r"^(found|finds|felt|feels|says|wants|liked|likes|praises|appreciated|appreciates|values|"
                       r"valued|enjoyed|notes|reports|dislikes|could not|never)\s+", re.I)


def clean_summary(summary: str) -> str:
    """Repetition guard for a small model: drop doubled words, and drop the second half of
    '<X> but <X>' when it only repeats the first half."""
    words = summary.split()
    words = [w for i, w in enumerate(words) if i == 0 or w.lower() != words[i - 1].lower()]
    text = " ".join(words)
    if " but " in text:
        first, second = text.split(" but ", 1)
        a = set(_words(_LEAD_VERB.sub("", first)))
        b = set(_words(_LEAD_VERB.sub("", second)))
        if b and len(a & b) / len(a | b) >= 0.7:
            text = first.rstrip(" ,.") + "."
    return text


def parse_output(raw: str, comment: str = "") -> FeedbackSummary:
    """'topic | sentiment | summary' -> FeedbackSummary. Anything malformed is kept but marked invalid."""
    text = raw.split("\n")[0]
    m = _OUT_RE.match(text)
    if m and m.group(1) in TOPICS and m.group(2) in SENTIMENTS:
        summary = clean_summary(m.group(3).strip())
        return FeedbackSummary(comment, m.group(1), m.group(2), summary, raw, True,
                               unsupported_words(summary, comment) if comment else [])
    return FeedbackSummary(comment, "unknown", "unknown", text.strip(), raw, False, [])


@torch.no_grad()
def generate(model, tok, prompts: list[list[int]], max_new_tokens: int = 40, batch_size: int = 8) -> list[str]:
    """Greedy decoding, batched with left padding. Returns the decoded continuations."""
    model.eval()
    eos, outs = tok.eos_token_id, []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        n = max(len(p) for p in chunk)
        ids = torch.tensor([[eos] * (n - len(p)) + p for p in chunk])
        mask = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in chunk])
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=eos, eos_token_id=eos)
        for row in gen[:, n:]:
            row = row.tolist()
            row = row[: row.index(eos)] if eos in row else row
            outs.append(tok.decode(row, skip_special_tokens=True))
    return outs


class Summarizer:
    """Model + tokenizer -> structured summaries. Works with a merged model or a LoRA-wrapped one."""

    def __init__(self, model, tok, max_new_tokens: int = 40, batch_size: int = 8, max_prompt_tokens: int = 160):
        self.model, self.tok = model, tok
        self.max_new_tokens, self.batch_size, self.max_prompt_tokens = max_new_tokens, batch_size, max_prompt_tokens

    @classmethod
    def from_adapter(cls, path: Path = DEFAULT_ADAPTER, merge: bool = True, **kw) -> "Summarizer":
        model, tok, _ = load_adapter(path, merge=merge)
        return cls(model, tok, **kw)

    @classmethod
    def from_merged(cls, directory: Path, **kw) -> "Summarizer":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return cls(AutoModelForCausalLM.from_pretrained(directory).eval(), AutoTokenizer.from_pretrained(directory), **kw)

    def raw(self, comments: list[str]) -> list[str]:
        prompts = [prompt_ids(self.tok, c, self.max_prompt_tokens) for c in comments]
        return generate(self.model, self.tok, prompts, self.max_new_tokens, self.batch_size)

    def summarize(self, comments: list[str], redact_pii: bool = True) -> list[FeedbackSummary]:
        cleaned = [redact(c) if redact_pii else (c, 0) for c in comments]
        results = []
        for (text, n), raw in zip(cleaned, self.raw([t for t, _ in cleaned])):
            r = parse_output(raw, text)
            r.redactions = n
            results.append(r)
        return results


def build_digest(results: list[FeedbackSummary]) -> dict:
    """Course-level view: sentiment split, topic counts, action items (complaints grouped by topic, the
    most-raised topic first) and strengths."""
    valid = [r for r in results if r.valid]
    by_topic: dict[str, Counter] = defaultdict(Counter)
    for r in valid:
        by_topic[r.topic][r.sentiment] += 1
    issues, strengths = defaultdict(list), defaultdict(list)
    for r in valid:
        if r.sentiment in ("negative", "mixed"):
            issues[r.topic].append(r.summary)
        else:
            strengths[r.topic].append(r.summary)
    rank = lambda d: sorted(d.items(), key=lambda kv: (-len(kv[1]), TOPICS.index(kv[0])))  # noqa: E731
    sent = Counter(r.sentiment for r in valid)
    return {
        "n_comments": len(results), "n_valid": len(valid),
        "sentiment": {s: sent.get(s, 0) for s in SENTIMENTS},
        "topics": {t: dict(by_topic[t]) for t in TOPICS if t in by_topic},
        "action_items": [{"topic": t, "count": len(v), "examples": v[:3]} for t, v in rank(issues)],
        "strengths": [{"topic": t, "count": len(v), "examples": v[:2]} for t, v in rank(strengths)],
        "needs_review": [r.comment for r in results if not r.valid or len(r.unsupported) >= 3],
    }


def digest_markdown(d: dict, title: str = "Course feedback digest") -> str:
    n = max(d["n_valid"], 1)
    s = d["sentiment"]
    lines = [f"# {title}", "",
             f"{d['n_comments']} comments: {s['negative']} negative ({s['negative'] / n:.0%}), "
             f"{s['mixed']} mixed, {s['positive']} positive.", "", "## Action items (most raised first)", ""]
    for i, a in enumerate(d["action_items"], 1):
        lines.append(f"{i}. **{a['topic']}** ({a['count']} student{'s' if a['count'] != 1 else ''})")
        lines += [f"   - {e}" for e in a["examples"]]
    if not d["action_items"]:
        lines.append("None raised.")
    lines += ["", "## What is working", ""]
    lines += [f"- **{a['topic']}** ({a['count']}): {a['examples'][0]}" for a in d["strengths"]] or ["Nothing yet."]
    if d["needs_review"]:
        lines += ["", f"## Read these yourself ({len(d['needs_review'])})", ""]
        lines += [f"- {c}" for c in d["needs_review"]]
    return "\n".join(lines) + "\n"


# =====================================================================================================
# --- Milestone 4: Evaluate the quality of generated summaries against original feedback.
# =====================================================================================================
# Against the ground-truth summaries: ROUGE-1/2/L F1 (written here, no extra dependency), topic and
# sentiment accuracy, the share of well-formed outputs. Against the original comment: how many summary
# words the comment does not support, and how much shorter the summary is. A keyword baseline (topic and
# sentiment by keyword lists, first 20 words of the comment as the "summary") is the bar to beat.

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _f1(overlap: int, n_hyp: int, n_ref: int) -> float:
    if not overlap or not n_hyp or not n_ref:
        return 0.0
    p, r = overlap / n_hyp, overlap / n_ref
    return 2 * p * r / (p + r)


def rouge_n(ref: str, hyp: str, n: int = 1) -> float:
    r, h = _ngrams(_words(ref), n), _ngrams(_words(hyp), n)
    return _f1(sum((r & h).values()), sum(h.values()), sum(r.values()))


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(ref: str, hyp: str) -> float:
    r, h = _words(ref), _words(hyp)
    return _f1(_lcs(r, h), len(h), len(r))


TOPIC_KEYWORDS = {
    "pacing": ["pace", "fast", "rushed", "rush", "slow", "speed", "crammed", "quickly", "behind"],
    "workload": ["workload", "hours", "heavy", "too much", "deadlines", "load", "busy", "time-consuming"],
    "lectures": ["lecture", "lecturer", "professor explained", "explain", "boring", "engaging", "talk"],
    "assignments": ["assignment", "homework", "project", "problem set", "starter", "instructions"],
    "assessment": ["exam", "midterm", "final", "quiz", "grade", "grading", "rubric", "marks", "test"],
    "materials": ["slides", "textbook", "notes", "reading", "videos", "book", "cheat sheet"],
    "support": ["ta", "tas", "office hours", "email", "forum", "help", "reply", "respond"],
    "technology": ["moodle", "canvas", "gradescope", "jupyter", "website", "platform", "software", "audio",
                   "crash", "lab computers", "blackboard", "online", "upload"],
}
POS_WORDS = set("great good clear helpful useful fair engaging excellent love loved liked enjoyed manageable easy "
                "patient amazing best interesting well thank thanks awesome right helped".split())
NEG_WORDS = set("bad confusing hard too unclear heavy late unanswered crashing crashed terrible poor never "
                "inconsistent outdated broken missing frustrating exhausting rushed lost boring difficult "
                "unrealistic wrong errors no lacked worst annoying".split())
CONTRAST = ("but", "however", "although", "though", "while", "on the other hand", "sadly", "to be fair")


def keyword_baseline(comment: str) -> FeedbackSummary:
    """No model at all: keyword topic, lexicon sentiment, the first 20 words as the summary."""
    low = " " + " ".join(_words(comment)) + " "
    scores = {t: sum(low.count(f" {k} ") + low.count(f" {k}s ") for k in kws) for t, kws in TOPIC_KEYWORDS.items()}
    topic = max(TOPICS, key=lambda t: (scores[t], -TOPICS.index(t)))
    ws = set(_words(comment))
    pos, neg = len(ws & POS_WORDS), len(ws & NEG_WORDS)
    contrast = any(f" {c} " in low for c in CONTRAST)
    sentiment = "mixed" if pos and neg and contrast else ("negative" if neg >= pos and neg else "positive")
    summary = " ".join(re.split(r"(?<=[.!?])\s", comment.strip())[0].split()[:20])
    return FeedbackSummary(comment, topic, sentiment, summary, "", True, [])


def evaluate(preds: list[FeedbackSummary], gold: list[Example]) -> dict:
    """Quality numbers for predictions aligned with gold examples (fractions 0..1, except word counts)."""
    n = max(len(gold), 1)
    pairs = list(zip(preds, gold))
    out = {
        "n": len(gold),
        "valid_format": sum(p.valid for p in preds) / n,
        "topic_acc": sum(p.topic == g.topic for p, g in pairs) / n,
        "sentiment_acc": sum(p.sentiment == g.sentiment for p, g in pairs) / n,
        # what the digest relies on: is this comment raising an issue (negative or mixed) or not?
        "issue_acc": sum((p.sentiment in ("negative", "mixed")) == (g.sentiment in ("negative", "mixed"))
                         for p, g in pairs) / n,
        "rouge1": sum(rouge_n(g.summary, p.summary, 1) for p, g in pairs) / n,
        "rouge2": sum(rouge_n(g.summary, p.summary, 2) for p, g in pairs) / n,
        "rougeL": sum(rouge_l(g.summary, p.summary) for p, g in pairs) / n,
        "summary_words": sum(len(p.summary.split()) for p in preds) / n,
        "compression": sum(len(p.summary.split()) / max(len(g.comment.split()), 1) for p, g in pairs) / n,
    }
    content = [(len(unsupported_words(p.summary, g.comment)),
                len([w for w in _words(p.summary) if w not in _STOP and w not in _SUMMARY_VERBS and not w.isdigit()]))
               for p, g in pairs]
    out["supported_words"] = 1 - sum(u for u, _ in content) / max(sum(c for _, c in content), 1)
    return out


def confusion(preds: list[FeedbackSummary], gold: list[Example], attr: str = "topic") -> dict:
    labels = TOPICS if attr == "topic" else SENTIMENTS
    m = {g: Counter() for g in labels}
    for p, g in zip(preds, gold):
        m[getattr(g, attr)][getattr(p, attr)] += 1
    return {g: dict(c) for g, c in m.items()}


def gold_as_predictions(gold: list[Example]) -> list[FeedbackSummary]:
    """The reference summaries scored like predictions (an upper bound for the faithfulness check)."""
    return [FeedbackSummary(g.comment, g.topic, g.sentiment, g.summary) for g in gold]


# =====================================================================================================
# --- Milestone 5: Implement adapter merging for inference.
# =====================================================================================================
# W' = W + (alpha / r) B @ A (transposed for GPT-2's Conv1D). After merging the model is a plain GPT-2
# again: same size as the base, no LoRA code needed to load it, and one matmul per layer instead of three.

def merge_lora(model: nn.Module) -> int:
    """Replace every LoRALinear by its base layer with the update baked in. Returns how many."""
    names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name in names:
        _replace(model, name, model.get_submodule(name).merged())
    return len(names)


def save_merged(model, tok, directory: Path) -> int:
    """Merge (if needed) and write a standard Hugging Face folder. Returns its size in bytes."""
    merge_lora(model)
    directory = Path(directory)
    model.save_pretrained(directory)
    tok.save_pretrained(directory)
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


@torch.no_grad()
def max_logit_diff(a, b, ids: torch.Tensor) -> float:
    return float((a(input_ids=ids).logits - b(input_ids=ids).logits).abs().max())


def time_generation(summarizer: Summarizer, comments: list[str], repeats: int = 1) -> float:
    """Milliseconds per comment for summarizer.raw (best of `repeats`)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        summarizer.raw(comments)
        best = min(best, time.perf_counter() - t0)
    return 1000 * best / max(len(comments), 1)
