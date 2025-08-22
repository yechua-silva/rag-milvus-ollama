import sys
from config import AppConfig
from src.infrastructure.document_loader import PdfDocumentLoader
from src.infrastructure.text_processor import SmartChunker
from src.infrastructure.embedding_manager import OllamaEmbeddingManager
from src.infrastructure.vector_store_manager import MilvusManager
from src.application.ingestion_service import IngestionService
from src.application.chat_service import ChatService


def main():
    config = AppConfig()

    # --- Inyección de Dependencias ---
    loader = PdfDocumentLoader("D:\\Proyectos\\LLM-RAG\\docs")
    chunker = SmartChunker(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
    embedder = OllamaEmbeddingManager(config.EMBEDDING_MODEL)

    # Para el vector store, necesitamos la dimensión del embedding
    test_embedding = embedder.get_embedding("test")
    vector_store = MilvusManager(
        uri=config.MILVUS_URI, collection_name=config.COLLECTION_NAME, embedding_dim=len(test_embedding)
    )

    # Creamos los "orquestadores"
    ingestion_service = IngestionService(loader, chunker, embedder, vector_store)
    chat_service = ChatService(embedder, vector_store, config.LLM_MODEL, config.SEARCH_TOP_K)

    # --- Lógica de Ejecución ---
    if "--ingest" in sys.argv:
        print("Iniciando proceso de ingesta para todos los documentos...")
        ingestion_service.ingest()
        print("Ingesta completada.")
    else:
        print("Sistema de Chat RAG listo. Escribe 'salir' para terminar.")
        while True:
            question = input("\nPregunta: ")
            if question.lower() == "salir":
                break
            answer = chat_service.ask(question)
            print(f"\nRespuesta:\n{answer}")


if __name__ == "__main__":
    main()
