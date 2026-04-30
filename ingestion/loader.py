"""PDF loader. Returns per-page text + structural metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class LoadedDoc:
    source_path: str
    title: str
    pages: list[PageText] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl",
}


def _clean(text: str) -> str:
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_TITLE_OVERRIDES = {
    "01_attention_is_all_you_need": "Attention Is All You Need (Vaswani et al., 2017)",
    "02_bert": "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
    "03_gpt3": "Language Models are Few-Shot Learners (GPT-3, Brown et al., 2020)",
    "04_ddpm": "Denoising Diffusion Probabilistic Models (Ho et al., 2020)",
    "05_ddim": "Denoising Diffusion Implicit Models (Song et al., 2020)",
    "06_rag_original": "Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis et al., 2020)",
    "07_rag_survey": "Retrieval-Augmented Generation for LLMs: A Survey (Gao et al., 2023)",
    "08_self_rag": "Self-RAG: Learning to Retrieve, Generate, and Critique (Asai et al., 2023)",
    "09_hyde": "Precise Zero-Shot Dense Retrieval with HyDE (Gao et al., 2022)",
    "10_vit": "An Image is Worth 16x16 Words (Vision Transformer, Dosovitskiy et al., 2020)",
    "11_clip": "Learning Transferable Visual Models from Natural Language Supervision (CLIP, Radford et al., 2021)",
    "12_react": "ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)",
    "13_chain_of_thought": "Chain-of-Thought Prompting Elicits Reasoning (Wei et al., 2022)",
    "14_llm_survey": "A Survey of Large Language Models (Zhao et al., 2023)",
}


def load_pdf(path: str | Path) -> LoadedDoc:
    path = Path(path)
    doc = pymupdf.open(path)
    pages: list[PageText] = []
    for i, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        cleaned = _clean(raw)
        if cleaned:
            pages.append(PageText(page_number=i, text=cleaned))
    title = _TITLE_OVERRIDES.get(path.stem) or _guess_title(pages, fallback=path.stem)
    doc.close()
    return LoadedDoc(source_path=str(path), title=title, pages=pages)


def _guess_title(pages: list[PageText], fallback: str) -> str:
    if not pages:
        return fallback
    first = pages[0].text
    for line in first.splitlines():
        line = line.strip()
        if 10 < len(line) < 180 and not line.lower().startswith(("abstract", "arxiv:")):
            return line
    return fallback


def discover_pdfs(papers_dir: str | Path) -> list[Path]:
    return sorted(Path(papers_dir).glob("*.pdf"))
