from sentence_transformers import SentenceTransformer
import settings

_model_cache = {}


def get_embedder(model_name: str = None, device: str = None):
    model_name = model_name or settings.EMBEDDING_MODEL
    device = device or settings.EMBEDDING_DEVICE or "cpu"
    cache_key = f"{model_name}:{device}"
    if cache_key not in _model_cache:
        _model_cache[cache_key] = SentenceTransformer(model_name, device=device)
    return _model_cache[cache_key]


def embed_texts(texts: list[str], model_name: str = None, device: str = None) -> list[list[float]]:
    embedder = get_embedder(model_name, device)
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


def embed_query(query: str, model_name: str = None, device: str = None) -> list[float]:
    return embed_texts([query], model_name, device)[0]
