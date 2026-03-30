from pathlib import Path
import chromadb
from chromadb.config import Settings
from rag.embedder import embed_query


class Retriever:
    def __init__(self, chroma_path: str, collection_name: str = "chunks"):
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve top_k chunks for a query."""
        embedding = embed_query(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # distance = 1 - cosine_similarity (Chroma uses cosine by default)
                # Convert to similarity score (higher = better)
                dist = results["distances"][0][i] if results["distances"] else 0.0
                score = round(1.0 - dist, 4)

                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": results["metadatas"][0][i].get("doc_id", ""),
                    "text": results["documents"][0][i],
                    "score": score,
                })
        return chunks

    def count(self) -> int:
        return self.collection.count()

    def exists(self) -> bool:
        try:
            return self.count() > 0
        except Exception:
            return False
