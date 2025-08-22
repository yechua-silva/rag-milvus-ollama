import os


# config.py
class AppConfig:
    """
    Centraliza las configuraciones de la aplicación.
    """

    # --- Rutas de Archivos ---
    DOCS_FOLDER = os.environ.get("DOCS_FOLDER", "./docs")  # Nueva configuración

    # --- Configuración de Milvus ---
    MILVUS_URI = os.environ.get("MILVUS_URI", "http://127.0.0.1:19530")
    COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "pdf_knowledge_base")

    # --- Configuración de Modelos ---
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "mxbai-embed-large")
    LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")

    # --- Configuración de Procesamiento de Texto ---
    CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

    # --- Configuración de Búsqueda ---
    SEARCH_TOP_K = int(os.environ.get("SEARCH_TOP_K", "5"))
