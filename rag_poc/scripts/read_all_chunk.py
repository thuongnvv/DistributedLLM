"""
Read and export all indexed chunks for a single node.

Usage:
	python scripts/read_all_chunk.py --node nd010283a
	python scripts/read_all_chunk.py --node nd010283a --format txt
	python scripts/read_all_chunk.py --node nd010283a --output /tmp/nd010283a_chunks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings

sys.path.insert(0, str(Path(__file__).parent.parent))


SCRIPT_DIR = Path(__file__).parent.parent
REGISTRY_PATH = SCRIPT_DIR / "data" / "providers" / "nodes.json"


def load_nodes() -> list[dict]:
	if not REGISTRY_PATH.exists():
		return []
	return json.loads(REGISTRY_PATH.read_text(encoding="utf-8")).get("nodes", [])


def get_node_entry(node_id: str) -> dict | None:
	for node in load_nodes():
		if node.get("node_id") == node_id:
			return node
	return None


def read_all_chunks(node_id: str, chroma_path: Path, batch_size: int = 1000) -> list[dict]:
	client = chromadb.PersistentClient(
		path=str(chroma_path),
		settings=Settings(anonymized_telemetry=False),
	)

	collection_name = f"chunks_{node_id}"
	try:
		collection = client.get_collection(name=collection_name)
	except Exception as exc:
		raise RuntimeError(
			f"Collection '{collection_name}' not found in {chroma_path}. "
			"Please index this node first."
		) from exc

	total = collection.count()
	chunks: list[dict] = []
	offset = 0

	while offset < total:
		result = collection.get(
			include=["documents", "metadatas"],
			limit=batch_size,
			offset=offset,
		)

		ids = result.get("ids") or []
		docs = result.get("documents") or []
		metas = result.get("metadatas") or []

		for idx, chunk_id in enumerate(ids):
			meta = metas[idx] if idx < len(metas) and metas[idx] else {}
			text = docs[idx] if idx < len(docs) else ""
			chunks.append(
				{
					"chunk_id": chunk_id,
					"doc_id": meta.get("doc_id", ""),
					"text": text,
				}
			)

		offset += len(ids)
		if not ids:
			break

	chunks.sort(key=lambda x: x.get("chunk_id", ""))
	return chunks


def build_txt_output(node_id: str, chunks: list[dict]) -> str:
	lines = [
		f"node_id: {node_id}",
		f"total_chunks: {len(chunks)}",
		"",
	]

	for i, chunk in enumerate(chunks, start=1):
		lines.extend(
			[
				"=" * 80,
				f"[{i}/{len(chunks)}] {chunk['chunk_id']}",
				f"doc_id: {chunk['doc_id']}",
				"-" * 80,
				chunk["text"],
				"",
			]
		)

	return "\n".join(lines)


def main() -> int:
	parser = argparse.ArgumentParser(description="Export all chunks for a node")
	parser.add_argument("--node", required=True, help="Node id (example: nd010283a)")
	parser.add_argument(
		"--format",
		choices=["json", "txt"],
		default="json",
		help="Output format",
	)
	parser.add_argument(
		"--output",
		help="Output file path. Default: logs/chunks/<node>_chunks.<format>",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=1000,
		help="Batch size when reading from Chroma",
	)
	args = parser.parse_args()

	if args.batch_size <= 0:
		print("Error: --batch-size must be > 0")
		return 1

	node = get_node_entry(args.node)
	if node is None:
		print(f"Error: node '{args.node}' not found in {REGISTRY_PATH}")
		return 1

	chroma_path = SCRIPT_DIR / node["chroma_path"]
	if not chroma_path.exists():
		print(f"Error: Chroma path does not exist: {chroma_path}")
		return 1

	chunks = read_all_chunks(args.node, chroma_path=chroma_path, batch_size=args.batch_size)

	output_path = Path(args.output) if args.output else SCRIPT_DIR / "logs" / "chunks" / f"{args.node}_chunks.{args.format}"
	output_path.parent.mkdir(parents=True, exist_ok=True)

	if args.format == "json":
		output_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
	else:
		output_path.write_text(build_txt_output(args.node, chunks), encoding="utf-8")

	print(f"Exported {len(chunks)} chunks for node '{args.node}'")
	print(f"Output: {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
