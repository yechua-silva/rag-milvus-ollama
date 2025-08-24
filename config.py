import os


# config.py
class AppConfig:
    """
    Centraliza y gestiona todas las configuraciones de la aplicación RAG.

    Esta clase proporciona acceso unificado a todas las variables de configuración
    necesarias para el funcionamiento del sistema, con valores por defecto y
    soporte para variables de entorno.

    Attributes:
        EMBEDDING_ONNX_MODEL (str): Modelo de embeddings para ONNX
        EMBEDDING_BATCH_SIZE (int): Tamaño de lote para generación de embeddings
        NUM_WORKERS (int): Número de workers para procesamiento paralelo
        USE_GPU (bool): Flag para habilitar el uso de GPU
        DOCS_FOLDER (str): Ruta a la carpeta de documentos
        MILVUS_URI (str): URI de conexión a Milvus
        COLLECTION_NAME (str): Nombre de la colección en Milvus
        EMBEDDING_MODEL (str): Modelo de embeddings para Ollama
        LLM_MODEL (str): Modelo LLM para generación de respuestas
        CHUNK_SIZE (int): Tamaño de chunks para división de texto
        CHUNK_OVERLAP (int): Solapamiento entre chunks
        SEARCH_TOP_K (int): Número de resultados a retornar en búsquedas
    """

    # --- Configuración de GPU ---
    EMBEDDING_ONNX_MODEL = os.environ.get("EMBEDDING_ONNX_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "64"))
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
    USE_GPU = os.environ.get("USE_GPU", "true").lower() == "true"

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
    SEARCH_TOP_K = int(os.environ.get("SEARCH_TOP_K", "10"))
