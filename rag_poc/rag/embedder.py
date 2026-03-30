from sentence_transformers import SentenceTransformer
import settings

_model_cache = {}


def get_embedder(model_name: str = None):
    model_name = model_name or settings.EMBEDDING_MODEL
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_texts(texts: list[str], model_name: str = None) -> list[list[float]]:
    embedder = get_embedder(model_name)
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


def embed_query(query: str, model_name: str = None) -> list[float]:
    return embed_texts([query], model_name)[0]
