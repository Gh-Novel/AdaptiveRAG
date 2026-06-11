"""AdaptiveRAG — under-the-hood pipeline visualizer.

Run: streamlit run app.py
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# suppress harmless noise from Streamlit's torch inspector + ChromaDB posthog client
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import streamlit as st

from agent.critic import critique, refine_query
from agent.planner import plan
from agent.router import route
from agent.tools import image_retrieve_and_reason
from config import AGENT_CONFIG, EMBEDDING_CONFIG, HOSTED, LLM_CONFIG, PATHS, RETRIEVAL_CONFIG
from ingestion.embedder import embed_query
from ingestion.indexer import fetch_embeddings
from llm.client_factory import get_llm
from retrieval.dense import Hit, dense_search
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.reranker import rerank
from retrieval.sparse import sparse_search

st.set_page_config(page_title="AdaptiveRAG — Underhood", page_icon="🔬", layout="wide")

# ── LLM backend check — shown before anything else ──────────────────
if not os.environ.get("GROQ_API_KEY"):
    # Running without Groq — check if Ollama is reachable locally
    try:
        import requests as _req
        _req.get("http://localhost:11434/api/tags", timeout=2).raise_for_status()
        _ollama_ok = True
    except Exception:
        _ollama_ok = False
    if not _ollama_ok:
        st.error(
            "**No LLM backend found.**\n\n"
            "- **Running on Hugging Face?** Add your `GROQ_API_KEY` secret in "
            "Space Settings → Variables and secrets. Get a free key at "
            "[console.groq.com](https://console.groq.com).\n"
            "- **Running locally?** Start Ollama: `ollama serve`"
        )
        st.stop()

# ───────────────────────────── styling ──────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

  html, body, [class*="st-"], .stMarkdown { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
  code, pre, .mono { font-family: 'JetBrains Mono', 'SF Mono', Menlo, monospace !important; }
  #MainMenu, footer { visibility: hidden; }

  h1 { letter-spacing: -.02em; }
  .app-sub { color: #8B949E; font-size: .9rem; margin-top: -0.4rem; }
  .badge-row { margin: .35rem 0 1rem 0; }

  /* step header */
  .phase-card {
    border-left: 2px solid #3D7EFF;
    padding: .55rem .95rem;
    margin: 1.15rem 0 .55rem 0;
    background: linear-gradient(90deg, rgba(61,126,255,.07), transparent 65%);
    border-radius: 0 4px 4px 0;
  }
  .phase-num  { font-family: 'JetBrains Mono', monospace; font-size: .67rem;
                letter-spacing: .14em; color: #3D7EFF; font-weight: 600; }
  .phase-title{ font-weight: 600; font-size: .98rem; color: #E6EDF3; }
  .phase-sub  { color: #8B949E; font-size: .8rem; margin-top: .1rem; }

  /* iteration divider */
  .iter-rule { display: flex; align-items: center; gap: .9rem; margin: 1.8rem 0 .3rem; }
  .iter-rule:before, .iter-rule:after { content: ''; flex: 1; height: 1px; background: #21262D; }
  .iter-rule span { font-family: 'JetBrains Mono', monospace; font-size: .72rem;
                    letter-spacing: .18em; color: #8B949E; }

  /* pills (stage tags) */
  .pill { display: inline-block; padding: .13rem .5rem; border-radius: 4px;
          font-family: 'JetBrains Mono', monospace; font-size: .67rem;
          font-weight: 500; letter-spacing: .05em; text-transform: uppercase;
          margin-right: .45rem; border: 1px solid transparent; }
  .pill-blue   { color: #79B8FF; border-color: rgba(121,184,255,.35); background: rgba(56,114,200,.10); }
  .pill-green  { color: #56D364; border-color: rgba(86,211,100,.35);  background: rgba(46,160,67,.10); }
  .pill-purple { color: #D2A8FF; border-color: rgba(210,168,255,.35); background: rgba(137,87,229,.10); }
  .pill-amber  { color: #E3B341; border-color: rgba(227,179,65,.35);  background: rgba(187,128,9,.10); }
  .pill-red    { color: #FF7B72; border-color: rgba(255,123,114,.35); background: rgba(218,54,51,.10); }
  .pill-grey   { color: #8B949E; border-color: rgba(139,148,158,.30); background: rgba(110,118,129,.10); }

  /* status chips */
  .chip { display: inline-flex; align-items: center; gap: .45rem;
          font-size: .8rem; font-weight: 500; padding: .28rem .65rem;
          border-radius: 4px; border: 1px solid; margin-right: .5rem; }
  .chip .dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
  .chip-ok   { color: #3FB950; border-color: rgba(63,185,80,.35);  background: rgba(63,185,80,.07); }
  .chip-ok .dot   { background: #3FB950; }
  .chip-warn { color: #D29922; border-color: rgba(210,153,34,.35); background: rgba(210,153,34,.07); }
  .chip-warn .dot { background: #D29922; }
  .chip-err  { color: #F85149; border-color: rgba(248,81,73,.35);  background: rgba(248,81,73,.07); }
  .chip-err .dot  { background: #F85149; }
  .chip-label { color: #8B949E; font-size: .7rem; text-transform: uppercase;
                letter-spacing: .08em; margin-right: .3rem; }

  /* metric restyle */
  [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; }
  [data-testid="stMetricLabel"] { color: #8B949E; text-transform: uppercase;
                                  font-size: .68rem; letter-spacing: .08em; }

  /* retrieved chunk cards */
  .chunk-card {
    background: rgba(255,255,255,.02);
    border: 1px solid #21262D;
    border-radius: 5px; padding: .55rem .75rem; margin-bottom: .4rem;
    font-size: .82rem; line-height: 1.45;
  }
  .chunk-meta { color: #8B949E; font-size: .72rem; margin-bottom: .3rem;
                font-family: 'JetBrains Mono', monospace; }

  .mini-vec { font-family: 'JetBrains Mono', monospace; font-size: .68rem;
              color: #8B949E; word-break: break-all; }

  /* health bar */
  .health-wrap { margin: .6rem 0 .9rem 0; }
  .health-row  { display: flex; align-items: baseline; gap: .7rem; margin-bottom: .35rem; }
  .health-label{ color: #8B949E; font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; }
  .health-val  { font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 600; }
  .health-bar  { height: 5px; background: #21262D; border-radius: 3px; overflow: hidden; }
  .health-fill { height: 100%; border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)


def chip(label: str, ok: bool | None, text_ok: str = "pass", text_bad: str = "flagged",
         warn: bool = False) -> str:
    """Small status chip: green/amber/red dot + text."""
    if ok is None:
        cls, txt = "chip-warn", text_bad
    elif ok:
        cls, txt = "chip-ok", text_ok
    else:
        cls, txt = ("chip-warn" if warn else "chip-err"), text_bad
    return (f"<span class='chip-label'>{label}</span>"
            f"<span class='chip {cls}'><span class='dot'></span>{txt}</span>")


# ───────────────────────────── helpers ──────────────────────────────
@st.cache_resource
def _llm():
    return get_llm()


def _load_manifest() -> dict:
    p = PATHS["manifest_path"]
    return json.loads(p.read_text()) if p.exists() else {}


def phase_header(num: int, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"<div class='phase-card'><div class='phase-num'>STEP {num:02d}</div>"
        f"<div class='phase-title'>{title}</div>"
        f"<div class='phase-sub'>{subtitle}</div></div>",
        unsafe_allow_html=True,
    )


def hits_to_df(hits: list[Hit], score_label: str = "score") -> pd.DataFrame:
    rows = []
    for h in hits:
        title = h.metadata.get("title") or h.metadata.get("source_path", "?")
        short = title.split(" (")[0]
        if len(short) > 38:
            short = short[:35] + "…"
        label = f"{short} · p{h.metadata.get('page_start')} · {h.chunk_id.split('::')[-1]}"
        rows.append({"chunk": label, score_label: float(h.score), "chunk_id": h.chunk_id})
    return pd.DataFrame(rows)


def render_hits(hits: list[Hit], badge_class: str, label: str, max_chars: int = 220) -> None:
    if not hits:
        st.caption(f"_(no {label.lower()} hits)_")
        return
    for i, h in enumerate(hits, start=1):
        meta = h.metadata
        snippet = h.text[:max_chars].replace("\n", " ")
        if len(h.text) > max_chars:
            snippet += "…"
        st.markdown(
            f"<div class='chunk-card'>"
            f"<div class='chunk-meta'>"
            f"<span class='pill {badge_class}'>{label} #{i}</span>"
            f"score <b>{h.score:.3f}</b> · "
            f"{meta.get('title','?')} · p.{meta.get('page_start')}–{meta.get('page_end')} · "
            f"<code>{h.chunk_id}</code>"
            f"</div>{snippet}</div>",
            unsafe_allow_html=True,
        )


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def vector_space_plot(query_vec: list[float], fused_hits: list[Hit],
                      dense_ids: set[str], sparse_ids: set[str],
                      kept_ids: set[str]) -> None:
    if not fused_hits:
        st.caption("_(nothing to plot)_")
        return
    embs = fetch_embeddings([h.chunk_id for h in fused_hits])
    rows = []
    vecs = [np.array(query_vec, dtype=np.float32)]
    for h in fused_hits:
        v = embs.get(h.chunk_id)
        if v is None:
            continue
        vecs.append(np.array(v, dtype=np.float32))
        in_d, in_s = h.chunk_id in dense_ids, h.chunk_id in sparse_ids
        in_keep = h.chunk_id in kept_ids
        if in_d and in_s:
            color = "fused (both)"
        elif in_d:
            color = "dense only"
        elif in_s:
            color = "sparse only"
        else:
            color = "other"
        title = (h.metadata.get("title") or "?").split(" (")[0][:40]
        label = f"{title} · p{h.metadata.get('page_start')}"
        rows.append({"label": label, "color": color, "size": 90 if in_keep else 50})
    if len(vecs) < 3:
        st.caption("_(need at least 2 hits for a 2D projection)_")
        return
    proj = pca_2d(np.vstack(vecs))
    df = pd.DataFrame(
        [{"x": proj[0, 0], "y": proj[0, 1], "label": "🔎 your question",
          "color": "QUERY", "size": 220}]
        + [{"x": proj[i + 1, 0], "y": proj[i + 1, 1], **rows[i]}
           for i in range(len(rows))]
    )
    st.scatter_chart(
        df, x="x", y="y", color="color", size="size",
        height=380, use_container_width=True,
    )
    st.caption(
        "PCA projection of the query embedding + fused hit embeddings. "
        "Larger points survived cross-encoder reranking."
    )


def render_embedding_card(query: str, qv: list[float], dt: float) -> None:
    arr = np.array(qv, dtype=np.float32)
    cols = st.columns([1, 1, 1, 3])
    cols[0].metric("Model", EMBEDDING_CONFIG["model"].split("/")[-1])
    cols[1].metric("Dimensions", len(qv))
    cols[2].metric("L2 norm", f"{float(np.linalg.norm(arr)):.3f}")
    cols[3].metric("Embed time", f"{dt*1000:.0f} ms")
    st.caption(f"Question ({len(query)} chars, ~{len(query.split())} words):")
    st.code(query, language="text")
    st.caption("First 32 dimensions of the embedding vector:")
    st.bar_chart(pd.DataFrame({"value": arr[:32]}), height=140, use_container_width=True)
    preview = ", ".join(f"{x:+.3f}" for x in arr[:8]) + ", …"
    st.markdown(f"<span class='mini-vec'>vector[0:8] = [{preview}]</span>",
                unsafe_allow_html=True)


# ───────────────────────────── pipeline view ──────────────────────────────
def _render_healing_trace(healing_trace: list[dict], health_score: float) -> None:
    """Render the healing trace panel below the answer."""
    if health_score >= 80:
        color, label = "#3FB950", "healthy"
    elif health_score >= 60:
        color, label = "#D29922", "fair"
    else:
        color, label = "#F85149", "degraded"
    st.markdown(
        f"<div class='health-wrap'>"
        f"<div class='health-row'>"
        f"<span class='health-label'>Answer health</span>"
        f"<span class='health-val' style='color:{color};'>{health_score:.0f}<span "
        f"style='color:#8B949E;font-size:.8rem;'> / 100 · {label}</span></span>"
        f"</div>"
        f"<div class='health-bar'><div class='health-fill' "
        f"style='width:{health_score:.0f}%;background:{color};'></div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if not healing_trace:
        st.caption("All checks passed on the first attempt — no healing required.")
        return
    for attempt in healing_trace:
        num = attempt.get("attempt", "?")
        healthy = attempt.get("healthy", False)
        issues = attempt.get("issues", [])
        label_str = ("all checks passed" if healthy
                     else "issues: " + ", ".join(i.replace("_", " ") for i in issues))
        with st.expander(f"Attempt {num} — {label_str}", expanded=not healthy):
            if healthy:
                st.caption("All detectors clear — answer accepted as-is.")
                continue
            st.markdown(
                chip("grounding", "hallucination" not in issues)
                + chip("chunk quality", "low_chunk_quality" not in issues, warn=True)
                + chip("coverage", "knowledge_gap" not in issues, text_bad="gap detected"),
                unsafe_allow_html=True,
            )
            st.markdown("")
            for action in attempt.get("actions", []):
                atype = action.get("type", "")
                detail = action.get("detail", "")
                st.markdown(
                    f"<div class='chunk-card'>"
                    f"<span class='pill pill-grey'>{atype.replace('_', ' ')}</span>"
                    f"{detail}</div>",
                    unsafe_allow_html=True,
                )
                if atype == "hallucination_fix":
                    for s in action.get("flagged_sentences", []):
                        st.caption(f"flagged: “{s[:120]}…”")


def visual_pipeline(query: str, enable_healing: bool = True) -> None:
    llm = _llm()

    # ── Step 1: embed the question ────────────────────────────────
    phase_header(1, "Question encoding",
                 "Convert text → 384-dim dense vector via sentence-transformers (MiniLM-L6).")
    t0 = time.time()
    qv = embed_query(query)
    render_embedding_card(query, qv, time.time() - t0)

    # ── Step 2: Self-RAG router ────────────────────────────────
    phase_header(2, "Self-RAG router",
                 "Decide whether to RETRIEVE, ANSWER_DIRECTLY, or CLARIFY before touching the index.")
    t0 = time.time()
    decision = route(query, llm=llm)
    dt = time.time() - t0
    pill_map = {"RETRIEVE": "pill-blue", "ANSWER_DIRECTLY": "pill-green", "CLARIFY": "pill-amber"}
    pill = pill_map.get(decision["action"], "pill-grey")
    st.markdown(
        f"<span class='pill {pill}'>{decision['action']}</span>"
        f"<span style='color:#9aa3b2;'>{decision.get('reason','')}</span>"
        f"<span style='float:right;color:#9aa3b2;font-size:.78rem;'>"
        f"router latency: {dt*1000:.0f} ms</span>",
        unsafe_allow_html=True,
    )

    if decision["action"] == "ANSWER_DIRECTLY":
        st.markdown("### Direct answer (no retrieval)")
        ans = llm.generate(prompt=query,
                           system="You are a helpful research assistant. Be concise.",
                           temperature=0.2)
        st.markdown(ans)
        return
    if decision["action"] == "CLARIFY":
        st.markdown("### Clarifying question")
        ans = llm.generate(
            prompt=("The user asked: " + query +
                    "\n\nIt is too ambiguous to answer well. Ask one short clarifying question."),
            system="You are a helpful research assistant.",
            temperature=0.2,
        )
        st.markdown(ans)
        return

    # ── Iterations of plan → retrieve → answer → critique ────────
    accumulated: list[Hit] = []
    current_query = query

    for it in range(AGENT_CONFIG["max_iterations"]):
        st.markdown(f"<div class='iter-rule'><span>ITERATION {it + 1}</span></div>",
                    unsafe_allow_html=True)
        if current_query != query:
            st.info(f"Refined query → **{current_query}**")

        # ── Step 3: plan ─────────────────────────────────────
        phase_header(3, "Planner", "LLM decomposes the question into focused sub-queries.")
        prior = ""
        if accumulated:
            titles = sorted({h.metadata.get("title", "?") for h in accumulated})
            prior = "Already gathered passages from: " + ", ".join(titles)
        t0 = time.time()
        steps = plan(current_query, prior_summary=prior, llm=llm)
        dt = time.time() - t0
        st.caption(f"Generated {len(steps)} sub-quer{'y' if len(steps)==1 else 'ies'} in {dt*1000:.0f} ms")
        for i, s in enumerate(steps, start=1):
            st.markdown(
                f"<div class='chunk-card'>"
                f"<span class='pill pill-purple'>sub-query {i}</span>"
                f"<b>{s['query']}</b>"
                f"<div class='chunk-meta' style='margin-top:.3rem;'>"
                f"rationale: {s.get('rationale','—')}</div></div>",
                unsafe_allow_html=True,
            )

        # ── Step 4: retrieval per sub-query ──────────────────
        phase_header(
            4,
            "Hybrid retrieval per sub-query",
            f"Dense (Chroma cosine, k={RETRIEVAL_CONFIG['dense_k']}) ∥ "
            f"Sparse (BM25, k={RETRIEVAL_CONFIG['sparse_k']}) → "
            f"Reciprocal Rank Fusion → Cross-encoder rerank "
            f"(BGE, top {RETRIEVAL_CONFIG['rerank_top_n']}).",
        )

        for si, step in enumerate(steps, start=1):
            with st.expander(f"Sub-query {si}: {step['query']}", expanded=(si == 1)):
                t0 = time.time()
                dense_hits = dense_search(step["query"])
                t_dense = time.time() - t0
                t0 = time.time()
                sparse_hits = sparse_search(step["query"])
                t_sparse = time.time() - t0
                t0 = time.time()
                fused = reciprocal_rank_fusion([dense_hits, sparse_hits],
                                               top_k=max(RETRIEVAL_CONFIG["dense_k"],
                                                         RETRIEVAL_CONFIG["sparse_k"]))
                t_fuse = time.time() - t0
                t0 = time.time()
                reranked = rerank(step["query"], fused)
                t_rerank = time.time() - t0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Dense hits", len(dense_hits), f"{t_dense*1000:.0f} ms")
                m2.metric("Sparse hits", len(sparse_hits), f"{t_sparse*1000:.0f} ms")
                m3.metric("After RRF", len(fused), f"{t_fuse*1000:.0f} ms")
                m4.metric("After rerank", len(reranked), f"{t_rerank*1000:.0f} ms")

                tabs = st.tabs([
                    "Dense · vectors",
                    "Sparse · FTS5 BM25",
                    "RRF fusion",
                    "Cross-encoder rerank",
                    "Vector space",
                ])
                with tabs[0]:
                    st.caption("Top-K nearest neighbors by cosine similarity.")
                    if dense_hits:
                        st.bar_chart(hits_to_df(dense_hits, "cosine_sim"),
                                     x="chunk", y="cosine_sim",
                                     height=260, use_container_width=True)
                    render_hits(dense_hits[:5], "pill-blue", "DENSE")

                with tabs[1]:
                    st.caption("Top-K BM25 keyword matches (normalized).")
                    if sparse_hits:
                        st.bar_chart(hits_to_df(sparse_hits, "bm25_norm"),
                                     x="chunk", y="bm25_norm",
                                     height=260, use_container_width=True)
                    render_hits(sparse_hits[:5], "pill-green", "BM25")

                with tabs[2]:
                    st.caption(
                        "Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank). "
                        "Combines dense + sparse rankings into one merged list."
                    )
                    if fused:
                        st.bar_chart(hits_to_df(fused[:12], "rrf_score"),
                                     x="chunk", y="rrf_score",
                                     height=280, use_container_width=True)
                    render_hits(fused[:5], "pill-purple", "FUSED")

                with tabs[3]:
                    st.caption(
                        "Cross-encoder scores (query, chunk) jointly — much more "
                        "accurate than bi-encoder cosine, but slower → only run on "
                        "the fused candidate set."
                    )
                    if reranked:
                        st.bar_chart(hits_to_df(reranked, "ce_score"),
                                     x="chunk", y="ce_score",
                                     height=240, use_container_width=True)
                    render_hits(reranked, "pill-amber", "RERANKED")

                with tabs[4]:
                    dense_ids = {h.chunk_id for h in dense_hits}
                    sparse_ids = {h.chunk_id for h in sparse_hits}
                    kept_ids = {h.chunk_id for h in reranked}
                    vector_space_plot(qv, fused[:20], dense_ids, sparse_ids, kept_ids)

                accumulated.extend(reranked)

        # ── Step 5: answer ─────────────────────────────────────
        # Dedupe + cap to 8 passages for the final prompt
        seen: set[str] = set()
        unique: list[Hit] = []
        for h in accumulated:
            if h.chunk_id in seen:
                continue
            seen.add(h.chunk_id)
            unique.append(h)
            if len(unique) >= 8:
                break
        context_lines, citations = [], []
        for i, h in enumerate(unique, start=1):
            meta = h.metadata
            head = (f"[{i}] {meta.get('title','?')} "
                    f"(p.{meta.get('page_start')}-{meta.get('page_end')})")
            context_lines.append(f"{head}\n{h.text}")
            citations.append({
                "n": i, "chunk_id": h.chunk_id,
                "title": meta.get("title"),
                "source_path": meta.get("source_path"),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "score": float(h.score),
            })
        context_block = "\n\n".join(context_lines)

        phase_header(5, "Context assembly + answer generation",
                     f"Top {len(unique)} unique passages → {LLM_CONFIG['model']} via {LLM_CONFIG['provider']}.")
        with st.expander("Context handed to the LLM", expanded=False):
            for c in citations:
                st.markdown(
                    f"**[{c['n']}]** {c['title']} · pages {c['page_start']}–{c['page_end']} · "
                    f"score `{c['score']:.3f}`"
                )
            st.code(context_block[:3000] + ("…" if len(context_block) > 3000 else ""),
                    language="text")

        t0 = time.time()
        ANSWER_SYSTEM = (
            "You are a careful research assistant. Use ONLY the provided passages to "
            "answer the question. Cite sources inline using the passage number in "
            "square brackets, e.g. [1] or [2][3] — never write the placeholder '[N]'. "
            "If the passages are insufficient, say so explicitly."
        )
        ANSWER_PROMPT = (
            f"Question: {query}\n\nPassages:\n{context_block}\n\n"
            "Write a concise, well-grounded answer. Use inline citations like [1], [2] "
            "that match the passage numbers above."
        )
        answer = llm.generate(prompt=ANSWER_PROMPT, system=ANSWER_SYSTEM, temperature=0.1)
        st.caption(f"LLM generation: {time.time()-t0:.1f} s")
        st.markdown("### Answer")
        st.markdown(answer)

        st.markdown("### Citations")
        for c in citations:
            st.markdown(
                f"**[{c['n']}]** {c['title']} — pages {c['page_start']}–{c['page_end']} "
                f"· score `{c['score']:.3f}` · `{Path(c['source_path']).name}`"
            )

        # ── Step 6: critic ─────────────────────────────────────
        phase_header(6, "Self-critique",
                     "LLM scores its own answer for grounding + completeness.")
        t0 = time.time()
        crit = critique(query, answer, context_block, llm=llm)
        conf_color = "#3FB950" if crit["confidence"] >= AGENT_CONFIG["confidence_threshold"] else "#D29922"
        st.markdown(
            chip("grounded", crit["grounded"], text_ok="yes", text_bad="no")
            + chip("complete", crit["complete"], text_ok="yes", text_bad="no", warn=True)
            + f"<span class='chip-label'>confidence</span>"
            f"<span class='chip' style='color:{conf_color};border-color:{conf_color}55;'>"
            f"{crit['confidence']:.2f} / {AGENT_CONFIG['confidence_threshold']:.2f}</span>",
            unsafe_allow_html=True,
        )
        if crit.get("missing"):
            st.warning(f"Missing: {crit['missing']}")
        st.caption(f"Critique latency: {time.time()-t0:.1f} s")

        if crit["confidence"] >= AGENT_CONFIG["confidence_threshold"] and crit["grounded"]:
            st.success(f"Confidence {crit['confidence']:.2f} ≥ threshold — answer accepted.")
            if enable_healing:
                phase_header(
                    7, "Self-Healing layer",
                    "Hallucination detection → chunk quality scoring → knowledge gap → regenerate if needed.",
                )
                with st.spinner("Running self-healing checks…"):
                    from healing.healing_loop import self_heal
                    healed = self_heal(query, answer, unique, citations, llm=llm)
                if healed.attempts_used > 0:
                    st.markdown("### Healed Answer")
                    st.markdown(healed.answer)
                    st.markdown("### Updated Citations")
                    for c in healed.citations:
                        st.markdown(
                            f"**[{c['n']}]** {c['title']} — "
                            f"pages {c.get('page_start')}–{c.get('page_end')} "
                            f"· score `{c['score']:.3f}`"
                        )
                _render_healing_trace(healed.healing_trace, healed.health_score)
            return

        if it < AGENT_CONFIG["max_iterations"] - 1:
            st.warning("Confidence below threshold — refining query and retrying.")
            current_query = refine_query(query, crit.get("missing", ""), llm=llm)
        else:
            st.error("Max iterations reached. Returning best-effort answer.")
            if enable_healing:
                phase_header(
                    7, "Self-Healing layer",
                    "Hallucination detection → chunk quality scoring → knowledge gap → regenerate if needed.",
                )
                with st.spinner("Running self-healing checks…"):
                    from healing.healing_loop import self_heal
                    healed = self_heal(query, answer, unique, citations, llm=llm)
                if healed.attempts_used > 0:
                    st.markdown("### Healed Answer")
                    st.markdown(healed.answer)
                _render_healing_trace(healed.healing_trace, healed.health_score)


# ───────────────────────────── sidebar + tabs ──────────────────────────────
def _sidebar() -> None:
    st.sidebar.title("AdaptiveRAG")
    st.sidebar.caption("Agentic + Self-RAG + Modular RAG")
    llm = _llm()
    ok = llm.health()
    backend = "Groq API" if HOSTED else "Ollama (local)"
    dot = "#3FB950" if ok else "#F85149"
    st.sidebar.markdown(
        f"**LLM backend**: <span style='display:inline-block;width:8px;height:8px;"
        f"border-radius:50%;background:{dot};margin:0 .25rem 1px 0;'></span>{backend}",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(f"**Model**: `{LLM_CONFIG['model']}`")
    st.sidebar.markdown(f"**Embedder**: `{EMBEDDING_CONFIG['model'].split('/')[-1]}`")
    st.sidebar.markdown(f"**Reranker**: `bge-reranker-base`")
    manifest = _load_manifest()
    if manifest:
        st.sidebar.markdown(f"**Index**: {manifest.get('n_chunks','?')} chunks across "
                            f"{len(manifest.get('chunks_per_doc',{}))} docs")
        with st.sidebar.expander("Documents"):
            for doc, n in sorted(manifest.get("chunks_per_doc", {}).items()):
                st.markdown(f"- `{doc}` — {n}")
    else:
        st.sidebar.warning("No index found. Run `python ingest.py --reset`.")
    st.sidebar.divider()
    st.sidebar.markdown("### Pipeline")
    st.sidebar.code(
        "question\n   ↓ embed (MiniLM)\n   ↓ Self-RAG router\n   ↓ planner → sub-queries\n"
        "   ↓ dense ∥ sparse\n   ↓ RRF fusion\n   ↓ cross-encoder rerank\n   ↓ LLM answer\n"
        "   ↓ self-critique → retry?\n   ↓ self-healing\n   → answer + citations",
        language="text",
    )


def pipeline_tab() -> None:
    st.subheader("Under the hood — every stage of the agentic RAG pipeline, live")
    st.caption(
        "Each step renders its inputs and outputs as it runs — embedding vector, "
        "router decision, planner sub-queries, dense vs sparse hits side-by-side, "
        "RRF fusion, cross-encoder rerank, vector-space projection, answer, self-critique."
    )
    samples = [
        "How does Self-RAG decide when to retrieve, and what reflection tokens does it use?",
        "Compare DDPM and DDIM sampling — what does DDIM gain by being non-Markovian?",
        "What is multi-head self-attention and why does parallelism matter?",
        "How does HyDE improve dense retrieval without relevance labels?",
        "How does ReAct combine reasoning and acting, vs chain-of-thought?",
        "hello, what can you do?",
    ]
    if "vq" not in st.session_state:
        st.session_state.vq = samples[0]
    cols = st.columns(3)
    for i, s in enumerate(samples):
        if cols[i % 3].button(s, key=f"vs{i}", use_container_width=True):
            st.session_state.vq = s
    q = st.text_area("Question", value=st.session_state.vq, height=80, key="vq_input")
    enable_healing = st.toggle(
        "Self-Healing layer",
        value=True,
        help="After the answer is generated, run hallucination detection, chunk quality "
             "scoring, and knowledge-gap checks — regenerating if issues are found.",
    )
    if st.button("Run pipeline", type="primary"):
        if q.strip():
            visual_pipeline(q.strip(), enable_healing=enable_healing)


def image_tab() -> None:
    st.subheader("Multimodal Q&A (Qwen3-VL)")
    st.caption(
        "Upload an image (e.g. a figure from a paper). Qwen3-VL captions it, the "
        "caption + question drives hybrid retrieval, then the model reasons over "
        "image + retrieved passages together."
    )
    uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
    q = st.text_input("Question about the image", "Explain what this figure shows.")
    go = st.button("Reason", type="primary", key="img_go")
    if uploaded:
        st.image(uploaded, width=400)
    if not (go and uploaded):
        return
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as f:
        f.write(uploaded.getbuffer())
        tmp_path = f.name
    try:
        with st.spinner("Captioning → retrieving → multimodal reasoning..."):
            out = image_retrieve_and_reason(tmp_path, q, llm=_llm())
        st.markdown("### Caption")
        st.write(out["caption"])
        st.markdown("### Answer")
        st.markdown(out["answer"])
        st.markdown("### Retrieved passages")
        for i, h in enumerate(out["hits"], start=1):
            st.markdown(
                f"**[{i}]** {h.metadata.get('title')} "
                f"(p.{h.metadata.get('page_start')}–{h.metadata.get('page_end')}) "
                f"· score `{h.score:.3f}`"
            )
            st.caption(h.text[:300] + ("…" if len(h.text) > 300 else ""))
    finally:
        os.unlink(tmp_path)


def kb_tab() -> None:
    """Knowledge Base Versioning tab."""
    st.subheader("Knowledge Base Versioning")
    st.caption(
        "Every ingest run creates a versioned snapshot of the index (kb_v1, kb_v2 …). "
        "Old snapshots are never deleted — rollback is a single metadata write. "
        "You can query any version independently and replay historical answers."
    )

    # ── lazy import so the tab loads even if versioning DB is empty ──
    try:
        from versioning.version_router import VersionRouter
        router = VersionRouter()
        history = router.list_versions()
    except Exception as exc:
        st.error(f"Could not initialise versioning layer: {exc}")
        return

    # ── current version banner ───────────────────────────────────────
    current = router.current_version()
    if current is None:
        st.warning(
            "No versioned snapshot found. Run `python ingest.py` to create v1."
        )
        return

    info = router.version_info(current)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current version", f"v{current}")
    c2.metric("Total snapshots", len(history))
    if info:
        c3.metric("Docs added (latest)", info.get("docs_added", 0))
        c4.metric("Docs changed (latest)", info.get("docs_changed", 0))

    st.divider()

    # ── version history table ────────────────────────────────────────
    st.markdown("### Version history")
    if history:
        df = pd.DataFrame(history)
        df["version"] = df["version"].apply(lambda v: f"v{v}")
        df["timestamp"] = df["timestamp"].str[:19].str.replace("T", " ")
        df = df.rename(columns={
            "version": "Version",
            "timestamp": "Created",
            "batch_name": "Batch",
            "docs_added": "Added",
            "docs_changed": "Changed",
            "docs_unchanged": "Unchanged",
            "reason": "Reason",
            "collection_name": "Collection",
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No versions yet.")

    st.divider()

    # ── rollback ─────────────────────────────────────────────────────
    st.markdown("### Rollback")
    st.caption(
        "Rolling back points the 'latest' pointer at a previous snapshot. "
        "The current snapshot is **not** deleted — you can roll forward again any time."
    )
    version_nums = sorted([v["version"] for v in history])
    if len(version_nums) > 1:
        target = st.selectbox(
            "Roll back to",
            options=[v for v in version_nums if v != current],
            format_func=lambda v: f"v{v}",
            key="kb_rollback_target",
        )
        if st.button("Roll back", type="secondary"):
            try:
                router.rollback(target)
                st.success(f"Rolled back to v{target}. Reload the page to see updated metrics.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    else:
        st.info("Need at least 2 versions to roll back.")

    st.divider()

    # ── per-version diff view ─────────────────────────────────────────
    st.markdown("### What changed per version")
    for v in history:
        vnum = v["version"]
        added = v.get("docs_added", 0)
        changed = v.get("docs_changed", 0)
        unchanged = v.get("docs_unchanged", 0)
        ts = (v.get("timestamp") or "")[:19].replace("T", " ")
        badge = "●" if vnum == current else "○"
        label = f"{badge} v{vnum}  —  {ts}  ·  +{added} added, ~{changed} changed, {unchanged} unchanged"
        with st.expander(label, expanded=(vnum == current)):
            cols = st.columns(4)
            cols[0].metric("Added", added)
            cols[1].metric("Changed", changed)
            cols[2].metric("Unchanged", unchanged)
            cols[3].metric("Reason", v.get("reason") or "—")
            st.caption(f"Collection: `{v.get('collection_name')}`  ·  Batch: `{v.get('batch_name')}`")
            # list docs active at this version
            try:
                from versioning.document_store import DocumentStore
                store = DocumentStore()
                docs = store.docs_at_version(vnum)
                if docs:
                    st.markdown("**Documents active at this version:**")
                    for d in sorted(docs, key=lambda x: x["doc_id"]):
                        status_icon = "●" if d["status"] == "active" else "○"
                        chk = (d.get("checksum") or "")[:12]
                        st.markdown(
                            f"{status_icon} `{d['doc_id']}` — "
                            f"{d.get('title','?')[:60]}  "
                            f"<span style='color:#9aa3b2;font-size:.75rem;'>"
                            f"sha256:{chk}…</span>",
                            unsafe_allow_html=True,
                        )
            except Exception:
                pass

    st.divider()

    # ── cross-version query ───────────────────────────────────────────
    st.markdown("### Query a specific version")
    st.caption(
        "Run the same question against different snapshots to see how the "
        "knowledge base evolution affects retrieval."
    )
    qv_text = st.text_input(
        "Question",
        value="How does Self-RAG decide when to retrieve?",
        key="kb_query_text",
    )
    col_v, col_k = st.columns([2, 1])
    qv_version = col_v.selectbox(
        "Version to query",
        options=["latest"] + [f"v{v}" for v in sorted(version_nums, reverse=True)],
        key="kb_query_version",
    )
    qv_k = col_k.slider("Top-K", 3, 10, 5, key="kb_query_k")

    if st.button("Run query", key="kb_query_btn"):
        version_arg: str | int = (
            "latest"
            if qv_version == "latest"
            else int(qv_version.lstrip("v"))
        )
        if not router.collection_exists(version_arg):
            st.error(
                f"ChromaDB collection for {qv_version} not found. "
                "The snapshot may exist in metadata but its collection was removed."
            )
        else:
            with st.spinner(f"Querying {qv_version}…"):
                try:
                    hits, resolved = router.query(
                        qv_text, version=version_arg, k=qv_k, log=True
                    )
                except Exception as exc:
                    st.error(str(exc))
                    hits, resolved = [], None

            if hits:
                st.success(f"Retrieved {len(hits)} passages from v{resolved}.")
                for i, h in enumerate(hits, start=1):
                    meta = h.metadata
                    title = meta.get("title", "?")
                    st.markdown(
                        f"<div class='chunk-card'>"
                        f"<div class='chunk-meta'>"
                        f"<span class='pill pill-purple'>#{i}</span>"
                        f"score <b>{h.score:.3f}</b> · {title} · "
                        f"p.{meta.get('page_start')}–{meta.get('page_end')}"
                        f"</div>{h.text[:300]}{'…' if len(h.text)>300 else ''}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("No hits returned.")

    st.divider()

    # ── query audit log ───────────────────────────────────────────────
    st.markdown("### Query audit log")
    st.caption("Every versioned query is recorded here for replay and debugging.")
    log = router.get_query_log(limit=20)
    if log:
        df_log = pd.DataFrame(log)
        df_log["timestamp"] = df_log["timestamp"].str[:19].str.replace("T", " ")
        df_log["version_used"] = df_log["version_used"].apply(
            lambda v: f"v{v}" if v is not None else "—"
        )
        df_log = df_log.rename(columns={
            "timestamp": "Time",
            "query": "Query",
            "version_used": "Version",
            "answer_hash": "Hash",
        })
        st.dataframe(df_log, use_container_width=True, hide_index=True)
    else:
        st.info("No queries logged yet.")


def main() -> None:
    _sidebar()
    st.title("AdaptiveRAG")
    manifest = _load_manifest()
    n_chunks = manifest.get("n_chunks", "—")
    n_docs = len(manifest.get("chunks_per_doc", {}))
    st.markdown(
        f"<div class='app-sub'>Agentic · Self-RAG · Modular — every pipeline stage "
        f"exposed live</div>"
        f"<div class='badge-row'>"
        f"<span class='pill pill-blue'>{LLM_CONFIG['provider']}</span>"
        f"<span class='pill pill-grey'>{LLM_CONFIG['model']}</span>"
        f"<span class='pill pill-purple'>{n_chunks} chunks</span>"
        f"<span class='pill pill-purple'>{n_docs} papers</span>"
        f"<span class='pill pill-green'>hybrid dense+FTS5</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    pipe, img, kb = st.tabs([
        "Pipeline",
        "Image Q&A",
        "Knowledge Base",
    ])
    with pipe:
        pipeline_tab()
    with img:
        image_tab()
    with kb:
        kb_tab()


if __name__ == "__main__":
    main()
