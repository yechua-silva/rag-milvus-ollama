import sys
from config import AppConfig
from src.infrastructure.document_loader import PdfDocumentLoader
from src.infrastructure.text_processor import BasicTextProcessor, SmartChunker
from src.infrastructure.embedding_manager import OllamaEmbeddingManager
from src.infrastructure.vector_store_manager import MilvusManager
from src.application.orchestrator import Orchestrator


def main():
    config = AppConfig()

    # --- Inyección de Dependencias ---
    text_processor = BasicTextProcessor()

    loader = PdfDocumentLoader("D:\\Proyectos\\LLM-RAG\\docs")

    chunker = SmartChunker(text_processor=text_processor, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)

    embedder = OllamaEmbeddingManager(config.EMBEDDING_MODEL)

    test_embedding = embedder.get_embedding("test")
    vector_store = MilvusManager(
        uri=config.MILVUS_URI, collection_name=config.COLLECTION_NAME, embedding_dim=len(test_embedding)
    )

    orchestrator = Orchestrator(
        loader=loader,
        text_processor=text_processor,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        llm_model=config.LLM_MODEL,
        search_top_k=config.SEARCH_TOP_K,
    )

    # --- Lógica de Ejecución ---
    if "--ingest" in sys.argv:
        print("Iniciando proceso de ingesta para todos los documentos...")
        orchestrator.ingest_documents()
        print("Ingesta completada.")
    else:
        print("Sistema de Chat RAG listo. Escribe 'salir' para terminar.")
        while True:
            question = input("\nPregunta: ")
            if question.lower() == "salir":
                break
            answer = orchestrator.ask_question(question)
            print(f"\nRespuesta:\n{answer}")


if __name__ == "__main__":
    main()
