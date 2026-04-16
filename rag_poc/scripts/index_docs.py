"""
Index documents into Chroma for each node.
Reads node list from data/providers/nodes.json.

Usage:
    python scripts/index_docs.py           # index all
    python scripts/index_docs.py --force  # reindex even if exists
    python scripts/index_docs.py --node node_a  # index specific node
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.node import RAGNode

SCRIPT_DIR = Path(__file__).parent.parent
REGISTRY_PATH = SCRIPT_DIR / "data" / "providers" / "nodes.json"


def load_nodes() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    return json.loads(REGISTRY_PATH.read_text()).get("nodes", [])


def build_rag_nodes(node_ids: list[str] | None = None) -> list[RAGNode]:
    """Build RAGNode list from registry, optionally filtered by node_ids."""
    nodes = []
    for n in load_nodes():
        if n.get("status") != "active":
            continue
        if node_ids and n["node_id"] not in node_ids:
            continue
        doc_path = SCRIPT_DIR / n["doc_path"]
        chroma_path = SCRIPT_DIR / n["chroma_path"]
        nodes.append(
            RAGNode(
                node_id=n["node_id"],
                doc_path=str(doc_path),
                chroma_path=str(chroma_path),
                domain=n.get("domain", "general"),
                style=n.get("style", "knowledge provider"),
                scope=n.get("scope", ""),
            )
        )
    return nodes


def main():
    parser = argparse.ArgumentParser(description="Index documents into Chroma")
    parser.add_argument("--force", action="store_true", help="Force reindex even if store exists")
    parser.add_argument("--node", help="Index specific node (e.g. node_a)")
    args = parser.parse_args()

    print("=" * 60)
    print("Document Indexer for RAG PoC")
    print("=" * 60)

    node_ids = [args.node] if args.node else None
    nodes = build_rag_nodes(node_ids)
    if not nodes:
        print("No nodes found. Check data/providers/nodes.json")
        return

    total = 0
    for node in nodes:
        print(f"\n[{node.node_id}] Indexing: {node.doc_path}")
        count = node.index_document(force=args.force)
        total += count

    print(f"\n{'=' * 60}")
    print(f"Done. Total chunks indexed: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
