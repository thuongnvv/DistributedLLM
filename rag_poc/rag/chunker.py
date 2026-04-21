from langchain_text_splitters import RecursiveCharacterTextSplitter
import settings


def create_chunker(chunk_size: int | None = None, chunk_overlap: int | None = None):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )


def chunk_document(text: str, doc_id: str, chunker=None):
    if chunker is None:
        chunker = create_chunker()

    raw_chunks = chunker.split_text(text)

    chunks = []
    for i, text in enumerate(raw_chunks):
        chunk_id = f"chunk_{i:05d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": text.strip(),
            }
        )
    return chunks
