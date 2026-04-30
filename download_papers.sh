#!/bin/bash
mkdir -p papers && cd papers

# Transformers
curl -L "https://arxiv.org/pdf/1706.03762" -o "01_attention_is_all_you_need.pdf"
curl -L "https://arxiv.org/pdf/1810.04805" -o "02_bert.pdf"
curl -L "https://arxiv.org/pdf/2005.14165" -o "03_gpt3.pdf"

# Diffusion
curl -L "https://arxiv.org/pdf/2006.11239" -o "04_ddpm.pdf"
curl -L "https://arxiv.org/pdf/2010.02502" -o "05_ddim.pdf"

# RAG
curl -L "https://arxiv.org/pdf/2005.11401" -o "06_rag_original.pdf"
curl -L "https://arxiv.org/pdf/2312.10997" -o "07_rag_survey.pdf"
curl -L "https://arxiv.org/pdf/2310.11511" -o "08_self_rag.pdf"
curl -L "https://arxiv.org/pdf/2212.10496" -o "09_hyde.pdf"

# Vision
curl -L "https://arxiv.org/pdf/2010.11929" -o "10_vit.pdf"
curl -L "https://arxiv.org/pdf/2103.00020" -o "11_clip.pdf"

# Agents
curl -L "https://arxiv.org/pdf/2210.03629" -o "12_react.pdf"
curl -L "https://arxiv.org/pdf/2201.11903" -o "13_chain_of_thought.pdf"
curl -L "https://arxiv.org/pdf/2303.18223" -o "14_llm_survey.pdf"

echo "Downloaded $(ls *.pdf | wc -l) papers"