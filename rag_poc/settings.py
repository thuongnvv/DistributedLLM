import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

# Auto-load .env from mvp/ sibling directory
_mvp_env = Path(__file__).parent.parent / "mvp" / ".env"
if _mvp_env.exists():
    from dotenv import load_dotenv
    load_dotenv(_mvp_env)

# RAG settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "10"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MAX_EVIDENCE_CHUNKS_PER_POINT = int(os.getenv("MAX_EVIDENCE_CHUNKS_PER_POINT", "3"))
MAX_EXTERNAL_POINTS_SYNTH = int(os.getenv("MAX_EXTERNAL_POINTS_SYNTH", "10"))
MAX_EXTERNAL_POINTS_GRADE = int(os.getenv("MAX_EXTERNAL_POINTS_GRADE", "8"))

# Node configs
NODE_A_DOC = os.getenv("NODE_A_DOC", str(BASE_DIR / "data/node_a/doc1_covid19.txt"))
NODE_A_CHROMA = os.getenv("NODE_A_CHROMA", str(BASE_DIR / "data/node_a/chroma_store/"))
NODE_A_ID = os.getenv("NODE_A_ID", "node_a")

NODE_B_DOC = os.getenv("NODE_B_DOC", str(BASE_DIR / "data/node_b/doc2_covid19.txt"))
NODE_B_CHROMA = os.getenv("NODE_B_CHROMA", str(BASE_DIR / "data/node_b/chroma_store/"))
NODE_B_ID = os.getenv("NODE_B_ID", "node_b")

NODE_C_DOC = os.getenv("NODE_C_DOC", str(BASE_DIR / "data/node_c/doc3_covid19.txt"))
NODE_C_CHROMA = os.getenv("NODE_C_CHROMA", str(BASE_DIR / "data/node_c/chroma_store/"))
NODE_C_ID = os.getenv("NODE_C_ID", "node_c")

NODE_D_DOC = os.getenv("NODE_D_DOC", str(BASE_DIR / "data/node_d/node_d_content.txt"))
NODE_D_CHROMA = os.getenv("NODE_D_CHROMA", str(BASE_DIR / "data/node_d/chroma_store/"))
NODE_D_ID = os.getenv("NODE_D_ID", "node_d")

NODE_E_DOC = os.getenv("NODE_E_DOC", str(BASE_DIR / "data/node_e/node_e_content.txt"))
NODE_E_CHROMA = os.getenv("NODE_E_CHROMA", str(BASE_DIR / "data/node_e/chroma_store/"))
NODE_E_ID = os.getenv("NODE_E_ID", "node_e")

NODE_F_DOC = os.getenv("NODE_F_DOC", str(BASE_DIR / "data/node_f/node_f_content.txt"))
NODE_F_CHROMA = os.getenv("NODE_F_CHROMA", str(BASE_DIR / "data/node_f/chroma_store/"))
NODE_F_ID = os.getenv("NODE_F_ID", "node_f")

# Orchestrator
K = int(os.getenv("K", "3"))
TAU_FAIL = int(os.getenv("TAU_FAIL", "2"))
WIN_BONUS = float(os.getenv("WIN_BONUS", "5.0"))
FAIL_PENALTY = float(os.getenv("FAIL_PENALTY", "2.0"))
POINT_SCORE_WEIGHT = float(os.getenv("POINT_SCORE_WEIGHT", "1.0"))
MAX_POINTS_PER_ANSWER = int(os.getenv("MAX_POINTS_PER_ANSWER", "5"))
MAX_USED_POINTS = int(os.getenv("MAX_USED_POINTS", "10"))

# LLM
LLM_MODE = os.getenv("LLM_MODE", "mega")  # mock, openai, mega
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "120.0"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "6"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MEGA_BASE_URL = os.getenv("MEGA_BASE_URL", "https://ai.megallm.io/v1")
MEGA_MODEL = os.getenv("MEGA_MODEL", "openai-gpt-oss-120b")
MEGALLM_API_KEY = os.getenv("MEGALLM_API_KEY", "")

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
API_BASE = os.getenv("API_BASE", f"http://localhost:{PORT}")

# Logs
LOGS_ROOT = os.getenv("LOGS_ROOT", str(BASE_DIR / "logs"))
