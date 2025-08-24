# main.py
import sys
from config import AppConfig
from src.infrastructure.document_loader import PdfDocumentLoader
from src.infrastructure.text_processor import BasicTextProcessor, SmartChunker
from src.infrastructure.vector_store_manager import MilvusManager
from src.application.orchestrator import Orchestrator


def main():
    """
    Función principal que inicia el sistema RAG.

    Configura e inicializa todos los componentes del sistema basándose
    en la configuración, y ejecuta el modo de ingesta o chat según
    los argumentos proporcionados.

    El modo de ingesta puede utilizar GPU (aceleración hardware) o CPU
    dependiendo de la configuración USE_GPU.
    """
    config = AppConfig()

    # --- Inyección de Dependencias ---
    text_processor = BasicTextProcessor()
    loader = PdfDocumentLoader(config.DOCS_FOLDER)
    chunker = SmartChunker(text_processor=text_processor, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)

    # Inicializamos el embedder que vamos a usar
    embedder = None
    if config.USE_GPU:
        print("Inicializando embedder en modo GPU...")
        from src.infrastructure.embedding_gpu import GPUEmbeddingGenerator

        embedder = GPUEmbeddingGenerator(config.EMBEDDING_ONNX_MODEL)
    else:
        print("Inicializando embedder en modo CPU (Ollama)...")
        from src.infrastructure.embedding_manager import OllamaEmbeddingManager

        embedder = OllamaEmbeddingManager(config.EMBEDDING_MODEL)

    # Obtenemos la dimensión del embedder que acabamos de crear.
    embedding_dim = embedder.get_embedding_dim()
    print(f"Dimensión de embedding detectada: {embedding_dim}")

    # **CORRECCIÓN**: Llamamos a MilvusManager con 'embedding_dim', como espera el constructor corregido.
    vector_store = MilvusManager(
        uri=config.MILVUS_URI, collection_name=config.COLLECTION_NAME, embedding_dim=embedding_dim
    )

    # --- Lógica de Ejecución ---
    # El resto del código ya es correcto y no necesita cambios.
    if "--ingest" in sys.argv:
        print("Iniciando proceso de ingesta...")
        orchestrator = Orchestrator(
            loader=loader,
            text_processor=text_processor,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            llm_model=config.LLM_MODEL,
            search_top_k=config.SEARCH_TOP_K,
        )
        orchestrator.ingest_documents()
        print("Ingesta completada.")
    else:
        chat_orchestrator = Orchestrator(
            loader=loader,
            text_processor=text_processor,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            llm_model=config.LLM_MODEL,
            search_top_k=config.SEARCH_TOP_K,
        )

        print("\nSistema de Chat RAG listo. Escribe 'salir' para terminar.")
        while True:
            question = input("\nPregunta: ")
            if question.lower() == "salir":
                break
            answer = chat_orchestrator.ask_question(question)
            print(f"\nRespuesta:\n{answer}")


if __name__ == "__main__":
    main()
