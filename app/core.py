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
