"""
One-time script to index documents into Chroma for each node.

Usage:
    python scripts/index_docs.py           # index all
    python scripts/index_docs.py --force  # reindex even if exists
    python scripts/index_docs.py --node node_a  # index specific node
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import settings
from rag.node import RAGNode


def build_rag_nodes() -> list[RAGNode]:
    return [
        RAGNode(
            node_id=settings.NODE_A_ID,
            doc_path=settings.NODE_A_DOC,
            chroma_path=settings.NODE_A_CHROMA,
            domain="medical_covid",
            style="medical researcher; evidence-based; cites sources",
            scope="COVID-19 disease, SARS-CoV-2 virus, symptoms, treatment, prevention, variants",
        ),
        RAGNode(
            node_id=settings.NODE_B_ID,
            doc_path=settings.NODE_B_DOC,
            chroma_path=settings.NODE_B_CHROMA,
            domain="medical_covid",
            style="healthcare provider; practical guidance; patient-focused",
            scope="COVID-19 disease, SARS-CoV-2 virus, symptoms, treatment, prevention, vaccination",
        ),
        RAGNode(
            node_id=settings.NODE_C_ID,
            doc_path=settings.NODE_C_DOC,
            chroma_path=settings.NODE_C_CHROMA,
            domain="medical_covid",
            style="public health official; WHO guidance; global perspective; informational",
            scope="COVID-19 disease, SARS-CoV-2 virus, symptoms, treatment, prevention, vaccination, variants, public health measures",
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Index documents into Chroma")
    parser.add_argument("--force", action="store_true", help="Force reindex even if store exists")
    parser.add_argument("--node", choices=["node_a", "node_b", "node_c", "all"], default="all")
    args = parser.parse_args()

    print("=" * 60)
    print("Document Indexer for RAG PoC")
    print("=" * 60)

    nodes = build_rag_nodes()
    total = 0

    for node in nodes:
        if args.node != "all" and node.node_id != args.node:
            continue

        print(f"\n[{node.node_id}] Indexing: {node.doc_path}")
        count = node.index_document(force=args.force)
        total += count

    print(f"\n{'=' * 60}")
    print(f"Done. Total chunks indexed: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
