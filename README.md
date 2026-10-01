# RAG Evaluation POC

A lightweight framework for evaluating Retrieval-Augmented Generation (RAG) systems. Test different chunking strategies, embedding models, and retrieval parameters to find the best configuration for your use case.

## Features

✅ **No external APIs** — runs entirely locally with Ollama  
✅ **Comprehensive metrics** — retrieval quality (Recall@k, MRR) + generation quality (correctness, faithfulness)  
✅ **Multiple configurations** — easily compare chunk size, overlap, and retrieval count  
✅ **LLM-based evaluation** — uses a stronger model to judge answer quality  
✅ **Reproducible results** — fixed seeds and deterministic evaluation  

## Quick Start

### Prerequisites

- **Ollama** installed and running ([download here](https://ollama.ai))
- **Python 3.12+** (or use `uv` to manage it)
- **uv** package manager ([optional but recommended](https://docs.astral.sh/uv/))

### 1. Install and download models

```bash
# Install Ollama and start the service
brew install ollama  # or download from ollama.ai
ollama serve

# In another terminal, download the models
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b
```

### 2. Clone and install dependencies

```bash
git clone <this-repo>
cd rag-eval
uv sync  # or: pip install -r requirements.txt
```

### 3. Run the evaluation

```bash
# Full run (50 questions, 4 configurations, ~1 hour)
uv run run.py

# Quick smoke test (6 questions)
uv run run.py --questions 6

# Retrieval metrics only, no LLM judge (faster)
uv run run.py --no-judge
```

## How It Works

### Architecture

1. **Data Loading** (`data.py`)
   - Loads SQuAD validation split (50 Wikipedia articles + questions)
   - Each question has reference answer and source location

2. **RAG Pipeline** (`rag.py`)
   - Chunks documents into overlapping windows
   - Embeds chunks using `nomic-embed-text`
   - Stores embeddings in Chroma vector database
   - Retrieves top-k chunks for each question
   - Generates answer using `qwen2.5:7b`

3. **Evaluation** (`metrics.py`)
   - **Retrieval metrics:** Recall@k (did you get the right chunk?), MRR (how early was it ranked?)
   - **Generation metrics:** correctness (matches reference?), faithfulness (claims grounded in context?)

4. **Comparison** (`run.py`)
   - Tests 4 configurations with different parameters
   - Generates results for each config
   - Prints comparison table
   - Runs sanity check on judge (scores deliberately wrong answers)

### Configurations Tested

| Config | Chunk Size | Overlap | Retrieve | Use Case |
|--------|-----------|---------|----------|----------|
| A | 50 words | 10 | k=3 | Fine-grained, many chunks |
| B | 150 words | 30 | k=3 | Balanced (often best) |
| C | 400 words | 50 | k=3 | Coarse, fewer chunks |
| D | 150 words | 30 | k=5 | Retrieve more context |

Edit the `CONFIGS` list in `run.py` to test different settings.

## Output

Results are saved to `results/` directory:

```
results/
├── chunk50_ov10_k3.jsonl    # Per-question results for config A
├── chunk150_ov30_k3.jsonl   # Per-question results for config B
├── chunk400_ov50_k3.jsonl   # Per-question results for config C
├── chunk150_ov30_k5.jsonl   # Per-question results for config D
└── summary.json             # Summary metrics for all configs
```

Each JSONL file contains:
- Question and reference answers
- System-generated answer
- Retrieved chunk IDs
- Retrieval metrics (recall@k, MRR)
- Generation metrics (correctness, faithfulness)
- Judge reasoning

## Example Results

```
config                      recall@k      mrr  contains_answer  correctness  faithfulness
─────────────────────────────────────────────────────────────────────────────────────────
chunk50_ov10_k3              0.85       0.68           0.60         0.75         0.90
chunk150_ov30_k3             0.88       0.70           0.65         0.78         0.92
chunk400_ov50_k3             0.90       0.72           0.70         0.80         0.91
chunk150_ov30_k5             0.85       0.65           0.62         0.76         0.89
```

## Customization

### Use your own documents

Replace `load_squad()` in `run.py` with your own data loader:

```python
docs = load_your_documents()  # List of Doc objects
questions = load_your_questions()  # List of Question objects
```

Your documents need:
- `id`, `title`, `text` fields
- Questions need: `id`, `doc_id`, `question`, `answers` (list), `answer_start`, `answer_end`

### Change models

Edit `rag.py`:
```python
EMBED_MODEL = "nomic-embed-text"  # Change to other Ollama models
GEN_MODEL = "qwen2.5:7b"          # Try: llama2, mistral, neural-chat, etc.
```

Edit `metrics.py`:
```python
JUDGE_MODEL = "qwen2.5:14b"       # Use larger models for more reliable judging
```

### Modify chunking strategy

Edit `run.py` `CONFIGS`:
```python
CONFIGS = [
    {"chunk_words": 100, "overlap_words": 20, "k": 3},
    {"chunk_words": 200, "overlap_words": 40, "k": 5},
    # ... add more
]
```

## Key Insights

1. **Larger chunks ≠ better.** A sweet spot exists around 150-200 words for most domains.

2. **The judge matters.** Small models (7B) give unreliable scores. Use 14B+ for evaluation.

3. **Separate retrieval from generation.** You can have perfect retrieval but poor generation (and vice versa).

4. **Overlap is important.** Without overlap, you lose context at chunk boundaries. But too much creates redundancy.

5. **Test multiple metrics.** Recall@k tells you if you found the right chunk. Correctness tells you if the answer is right. Both matter.

## Next Steps

- **Add a reranker** — retrieve k*2 chunks, rerank to top k
- **Test embedding models** — try `OpenAI`, `Voyage`, or other embeddings
- **Measure latency** — time retrieval + generation
- **A/B test in production** — validate eval results on real users
- **Fine-tune models** — train small model on your domain

## Metrics Explained

### Retrieval Metrics (no LLM)
- **Recall@k**: Binary. Did the correct chunk appear in top k? (0 or 1)
- **MRR**: Reciprocal rank of first correct chunk. Higher = found earlier. (0-1)

### Generation Metrics (LLM judge)
- **Correctness**: Does the system answer match the reference answer? (0-1, normalized from 0-2 scale)
- **Faithfulness**: Is every claim in the system answer grounded in retrieved context? (0-1, normalized)

## Troubleshooting

**"ollama: command not found"**
- Ollama is not in your PATH. Install it from ollama.ai or add to PATH.

**"Connection refused" to Ollama**
- Is Ollama running? Run `ollama serve` in another terminal.

**Models are very slow**
- Using a Mac with limited RAM? Test with smaller models: `qwen2.5:3b` or `mistral:7b`
- Reduce `n_questions` to 10-20 for faster iteration.

**Judge scores are all very high**
- Your judge model might be too small. Upgrade to `qwen2.5:14b` or use an API model.

## License

MIT

## Questions?

Feel free to open an issue or reach out. Would love to hear how you use this for your RAG evaluation!

---

**Built as a POC for rigorous RAG evaluation without complex infrastructure.**
