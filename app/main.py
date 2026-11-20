"""Streamlit entry point: `streamlit run app/main.py`

Upload survey comments -> one structured line per comment -> a course digest with ranked action items.
Instructors can rate each summary (1-5 stars); ratings are kept in data/app.json and shown per topic.
"""
import hashlib
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import core  # noqa: E402
from app.storage import Store  # noqa: E402

SAMPLE = ROOT / "samples" / "cs101_feedback.csv"
SENT_ICON = {"positive": "🟢", "mixed": "🟡", "negative": "🔴", "unknown": "⚪"}

st.set_page_config(page_title="Student Feedback Summarizer", layout="wide")
st.title("Student Feedback Summarizer")
st.write("Upload end-of-term comments. A LoRA-tuned GPT-2 turns each one into `topic | sentiment | summary`, "
         "then the digest ranks what to fix first by how many students raised it.")


@st.cache_resource(show_spinner="Loading distilgpt2 + LoRA adapter (first time downloads ~330 MB)...")
def load_summarizer(source: str):
    path = ROOT / source
    if path.is_dir():
        return core.Summarizer.from_merged(path)
    return core.Summarizer.from_adapter(path, merge=True)


def model_choices() -> list[str]:
    found = []
    if (ROOT / "models" / "merged").is_dir():
        found.append("models/merged")
    found += sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "models").glob("*.pt"))
    return found or [str(core.DEFAULT_ADAPTER)]


def key_for(comment: str) -> str:
    return "rating:" + hashlib.sha1(comment.encode()).hexdigest()[:12]


store = Store(os.environ.get("DATA_PATH") or str(ROOT / "data" / "app.json"))

with st.sidebar:
    st.header("Settings")
    source = st.selectbox("Model", model_choices(), help="A merged folder loads fastest; a .pt adapter is "
                          "merged into distilgpt2 on load.")
    redact_pii = st.toggle("Redact emails, phone numbers, IDs, names", value=True)
    st.caption("Redaction runs before the model sees the text, and the stored comments are the redacted ones.")

tab_run, tab_ratings = st.tabs(["Summarize", "Instructor ratings"])

with tab_run:
    col_up, col_paste = st.columns(2)
    with col_up:
        upload = st.file_uploader("Feedback file", type=["csv", "txt", "json", "jsonl"],
                                  help="CSV: a column named comment/feedback/response/text is used, otherwise "
                                       "the wordiest column. TXT: one comment per line or per paragraph.")
        use_sample = st.button("Use the sample CS101 survey", disabled=not SAMPLE.exists())
    with col_paste:
        pasted = st.text_area("...or paste comments, one per line", height=150)

    comments: list[str] = []
    if upload is not None:
        comments = core.read_feedback(upload.name, upload.getvalue())
    elif use_sample:
        comments = core.read_feedback(SAMPLE.name, SAMPLE.read_bytes())
        st.session_state["sample"] = True
    elif pasted.strip():
        comments = core.read_feedback("pasted.txt", pasted)
    elif st.session_state.get("sample"):
        comments = core.read_feedback(SAMPLE.name, SAMPLE.read_bytes())

    if comments:
        st.caption(f"{len(comments)} comments loaded.")
        if st.button("Summarize", type="primary"):
            with st.spinner(f"Summarizing {len(comments)} comments..."):
                st.session_state["results"] = load_summarizer(source).summarize(comments, redact_pii=redact_pii)
                st.session_state["source"] = source

    results = st.session_state.get("results")
    if results:
        d = core.build_digest(results)
        n = max(d["n_valid"], 1)
        m = st.columns(4)
        m[0].metric("Comments", d["n_comments"])
        m[1].metric("Negative", f"{d['sentiment']['negative'] / n:.0%}")
        m[2].metric("Mixed", f"{d['sentiment']['mixed'] / n:.0%}")
        m[3].metric("Top issue", d["action_items"][0]["topic"] if d["action_items"] else "none")

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Action items, most raised first")
            for i, a in enumerate(d["action_items"], 1):
                with st.expander(f"{i}. {a['topic']} ({a['count']})", expanded=i <= 3):
                    for e in a["examples"]:
                        st.write(f"- {e}")
            st.subheader("What is working")
            for a in d["strengths"]:
                st.write(f"**{a['topic']}** ({a['count']}): {a['examples'][0]}")
        with right:
            st.subheader("Comments by topic")
            chart = {t: {s: d["topics"].get(t, {}).get(s, 0) for s in core.SENTIMENTS} for t in core.TOPICS}
            st.bar_chart(chart, color=["#2e9e5b", "#d64545", "#e0b422"], horizontal=True)
            if d["needs_review"]:
                st.warning(f"{len(d['needs_review'])} comment(s) the model could not summarize cleanly: "
                           "read these yourself (listed in the digest download).")

        md = core.digest_markdown(d)
        st.download_button("Download digest (Markdown)", md, file_name="feedback_digest.md")

        st.subheader("Every comment")
        st.caption("Rate each summary: ratings are saved and summarized in the 'Instructor ratings' tab.")
        for i, r in enumerate(results):
            with st.container(border=True):
                a, b = st.columns([5, 1])
                a.markdown(f"{SENT_ICON.get(r.sentiment, '⚪')} **{r.topic}** · {r.sentiment} — {r.summary}")
                a.caption(r.comment + (f"  ·  {r.redactions} detail(s) redacted" if r.redactions else ""))
                if r.unsupported:
                    a.caption("Words not found in the comment: " + ", ".join(r.unsupported))
                with b:
                    stars = st.feedback("stars", key=f"stars-{i}-{key_for(r.comment)}")
                if stars is not None:
                    store.put(key_for(r.comment), {"comment": r.comment, "topic": r.topic, "sentiment": r.sentiment,
                                                   "summary": r.summary, "stars": stars + 1,
                                                   "model": st.session_state.get("source", source)})

with tab_ratings:
    ratings = [v for k, v in store.all().items() if k.startswith("rating:")]
    if not ratings:
        st.info("No ratings yet. Rate summaries on the Summarize tab.")
    else:
        avg = sum(r["stars"] for r in ratings) / len(ratings)
        c = st.columns(3)
        c[0].metric("Ratings", len(ratings))
        c[1].metric("Average", f"{avg:.2f} / 5")
        c[2].metric("Rated 4-5", f"{sum(r['stars'] >= 4 for r in ratings) / len(ratings):.0%}")
        by_topic: dict[str, list[int]] = {}
        for r in ratings:
            by_topic.setdefault(r["topic"], []).append(r["stars"])
        st.subheader("Average stars by topic")
        st.bar_chart({t: sum(v) / len(v) for t, v in sorted(by_topic.items())})
        st.subheader("Lowest-rated summaries (candidates for new training examples)")
        for r in sorted(ratings, key=lambda r: r["stars"])[:10]:
            st.write(f"{'★' * r['stars']}{'☆' * (5 - r['stars'])}  **{r['topic']}** — {r['summary']}")
            st.caption(r["comment"])
        st.download_button("Export ratings (JSON)", core.json.dumps(ratings, indent=2), file_name="ratings.json")
