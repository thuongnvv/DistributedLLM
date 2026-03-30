"""
RAGNode: A node that uses retrieval-augmented generation to answer and grade.

Each node has its own document, vector store, and persona.
Both Stage 1 (answer) and Stage 3 (grade) use RAG retrieval.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add repo root so we can import mvp modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataclasses import dataclass, field

from rag.chunker import chunk_document
from rag.embedder import embed_texts
from rag.retriever import Retriever
from models import Citation

import settings


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


class RAGNode:
    def __init__(
        self,
        node_id: str,
        doc_path: str,
        chroma_path: str,
        domain: str,
        style: str,
        scope: str,
        top_k: int = None,
    ):
        self.node_id = node_id
        self.doc_path = Path(doc_path)
        self.chroma_path = Path(chroma_path)
        self.domain = domain
        self.style = style
        self.scope = scope
        self.top_k = top_k or settings.TOP_K

        self._retriever: Retriever | None = None
        self._doc_id: str = ""

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(
                chroma_path=str(self.chroma_path),
                collection_name=f"chunks_{self.node_id}",
            )
        return self._retriever

    def index_document(self, force: bool = False) -> int:
        """Index the document into Chroma. Returns number of chunks."""
        # Reset retriever cache so we always get fresh collection reference
        self._retriever = None

        if self.retriever.exists() and not force:
            count = self.retriever.count()
            print(f"  [{self.node_id}] Chroma store already exists ({count} chunks), skipping. Use --force to reindex.")
            return count

        # Clear existing
        try:
            self.retriever.client.delete_collection(
                name=self.retriever.collection_name
            )
        except Exception:
            pass

        # Reset cache again after deletion so we get fresh collection
        self._retriever = None

        # Load document
        self._doc_id = self.doc_path.name
        text = self.doc_path.read_text(encoding="utf-8")

        # Chunk
        chunks = chunk_document(text, self._doc_id)
        print(f"  [{self.node_id}] Split into {len(chunks)} chunks")

        # Embed
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_texts(chunk_texts)

        # Store in Chroma
        collection = self.retriever.collection
        collection.add(
            ids=[c["chunk_id"] for c in chunks],
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=[{"doc_id": self._doc_id} for _ in chunks],
        )

        print(f"  [{self.node_id}] Indexed {len(chunks)} chunks into Chroma at {self.chroma_path}")
        return len(chunks)

    def retrieve(self, query: str, top_k: int = None) -> list[Citation]:
        """Retrieve top_k chunks for a query. Returns Citation list."""
        top_k = top_k or self.top_k
        raw = self.retriever.retrieve(query, top_k=top_k)
        return [
            Citation(
                doc_id=r["doc_id"],
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
            )
            for r in raw
        ]

    def build_rag_context(self, citations: list[Citation]) -> str:
        """Build context string from retrieved citations for LLM prompt."""
        if not citations:
            return ""
        parts = []
        for c in citations:
            parts.append(f"[{c.doc_id}:{c.chunk_id}]\n{c.text}")
        return "\n\n---\n\n".join(parts)

    def build_rag_prompt(
        self,
        query: str,
        citations: list[Citation],
        include_citation_rules: bool = True,
    ) -> str:
        """Build user prompt with RAG context + query."""
        context = self.build_rag_context(citations)
        citation_rules = (
            "\n- Cite sources inline as [doc_id:chunk_id] for every factual claim.\n"
            "- Answer ONLY using the provided context. If insufficient, say so.\n"
        ) if include_citation_rules else ""

        return (
            f"{context}\n\n"
            f"{citation_rules}"
            f"USER QUESTION: {query}\n\n"
            "ANSWER:"
        )
