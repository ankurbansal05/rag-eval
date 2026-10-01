"""The RAG pipeline under test: chunk -> embed -> retrieve -> generate."""

import re
from dataclasses import dataclass

import chromadb
import ollama

EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen2.5:7b"


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    start: int  # char span inside the source doc
    end: int


def chunk_doc(doc, chunk_words: int, overlap_words: int) -> list[Chunk]:
    """Split a doc into windows of `chunk_words` words, keeping char offsets."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", doc.text)]
    step = max(1, chunk_words - overlap_words)
    chunks = []
    for i in range(0, len(words), step):
        window = words[i : i + chunk_words]
        start, end = window[0][0], window[-1][1]
        chunks.append(Chunk(f"{doc.id}#{i}", doc.id, doc.text[start:end], start, end))
        if i + chunk_words >= len(words):
            break
    return chunks


def embed(texts: list[str], batch: int = 64) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), batch):
        out.extend(ollama.embed(model=EMBED_MODEL, input=texts[i : i + batch])["embeddings"])
    return out


class Index:
    def __init__(self, docs, chunk_words: int, overlap_words: int):
        self.chunks = [c for d in docs for c in chunk_doc(d, chunk_words, overlap_words)]
        self.by_id = {c.id: c for c in self.chunks}
        client = chromadb.EphemeralClient()
        name = f"c{chunk_words}_{overlap_words}"
        try:
            client.delete_collection(name)
        except Exception:
            pass
        self.col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        # nomic-embed-text expects task prefixes on documents and queries.
        vectors = embed([f"search_document: {c.text}" for c in self.chunks])
        self.col.add(ids=[c.id for c in self.chunks], embeddings=vectors)

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        [qvec] = embed([f"search_query: {query}"])
        res = self.col.query(query_embeddings=[qvec], n_results=k)
        return [self.by_id[i] for i in res["ids"][0]]


GEN_PROMPT = """Answer the question using only the context below.
Give a short, direct answer. If the context does not contain the answer, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""


def generate(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n---\n\n".join(c.text for c in chunks)
    resp = ollama.generate(
        model=GEN_MODEL,
        prompt=GEN_PROMPT.format(context=context, question=question),
        options={"temperature": 0},
    )
    return resp["response"].strip()
