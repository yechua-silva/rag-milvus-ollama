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
    ... (el resto de la documentación no cambia)
    """
    config = AppConfig()

    # --- Inyección de Dependencias --- (sin cambios aquí)
    text_processor = BasicTextProcessor()
    loader = PdfDocumentLoader(config.DOCS_FOLDER)
    chunker = SmartChunker(text_processor=text_processor, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)

    embedder = None
    if config.USE_GPU:
        print("Inicializando embedder en modo GPU...")
        from src.infrastructure.embedding_gpu import GPUEmbeddingGenerator

        embedder = GPUEmbeddingGenerator(config.EMBEDDING_ONNX_MODEL)
    else:
        print("Inicializando embedder en modo CPU (Ollama)...")
        from src.infrastructure.embedding_manager import OllamaEmbeddingManager

        embedder = OllamaEmbeddingManager(config.EMBEDDING_MODEL)

    embedding_dim = embedder.get_embedding_dim()
    print(f"Dimensión de embedding detectada: {embedding_dim}")

    vector_store = MilvusManager(
        uri=config.MILVUS_URI, collection_name=config.COLLECTION_NAME, embedding_dim=embedding_dim
    )

    # --- Lógica de Ejecución ---
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

            # --- CORRECCIÓN: Formateo de la respuesta ---
            response_obj = chat_orchestrator.ask_question(question)

            # 1. Imprimir la respuesta de texto del LLM
            print("\nRespuesta:")
            print(response_obj.answer)

            # 2. Imprimir las fuentes consultadas de forma clara
            if response_obj.source_chunks:
                print("\n--- Fuentes Consultadas ---")
                # Usamos un set para evitar mostrar la misma página múltiples veces
                sources = set()
                for result in response_obj.source_chunks:
                    source_file = result.chunk.metadata.get("source", "Desconocido")
                    page_num = result.chunk.metadata.get("page", "N/A")
                    sources.add(f"- Documento: {source_file}, Página: {page_num}")

                # Imprimir las fuentes únicas y ordenadas
                for source in sorted(list(sources)):
                    print(source)


if __name__ == "__main__":
    main()
