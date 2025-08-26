from pymilvus import MilvusClient
from typing import List
from tqdm import tqdm
from src.application.interfaces import VectorStore, Retriever
from src.domain.models import DocumentChunk, SearchResult


class MilvusManager(VectorStore, Retriever):
    """Sabe como interactuar con Milvus: configurar, insertar y buscar"""

    def __init__(self, uri: str, collection_name: str, embedding_dim: int):
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

    def set_collection(self):
        """Configura la colección de Milvus, asegurando la dimensión correcta."""
        if self.client.has_collection(collection_name=self.collection_name):
            print(f"Eliminando colección existente '{self.collection_name}' para asegurar consistencia de dimensión.")
            self.client.drop_collection(collection_name=self.collection_name)

        print(f"Creando nueva colección '{self.collection_name}' con dimensión {self.embedding_dim}...")
        self.client.create_collection(collection_name=self.collection_name, dimension=self.embedding_dim)
        print("Colección creada con éxito.")

    def insert(self, chunks: List[DocumentChunk], batch_size: int = 100):
        """Insertar chunks en Milvus con embeddings de manera eficiente"""

        # Adaptacion a uso de models
        data_to_insert = [
            {
                "id": hash(chunk.chunk_id),
                "vector": chunk.embedding,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
            }
            for chunk in chunks
            if chunk.embedding is not None
        ]

        if not data_to_insert:
            print("Advertencia: No hay chunks con emebeddings para insertar.")
            return

        print(f"Insertando {len(data_to_insert)} chunks en Milvus...")
        for i in tqdm(range(0, len(data_to_insert), batch_size), desc="Insertando lotes"):
            batch = data_to_insert[i : i + batch_size]
            try:
                self.client.insert(collection_name=self.collection_name, data=batch)
            except Exception as e:
                print(f"Error insertando lote {i // batch_size}: {e}")
                # Intentar insertar individualmente los elementos del lote con error
                for item in batch:
                    try:
                        self.client.insert(collection_name=self.collection_name, data=[item])
                    except Exception as single_error:
                        print(f"Error insertando item individual: {single_error}")
        try:
            self.client.compact(collection_name=self.collection_name)
            print("Datos persistidos con compact")
        except Exception as e:
            print(f"Error haciendo compact: {e}")

    def search(self, vector: List[float], top_k: int) -> List[SearchResult]:
        """Busca en la base de conocimiento"""
        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=top_k,
            output_fields=["text", "metadata", "chunk_id", "doc_id"],
        )

        results = []
        for res in search_res[0]:
            retrieved_chunk = DocumentChunk(
                chunk_id=res["entity"]["chunk_id"],
                doc_id=res["entity"]["doc_id"],
                text=res["entity"]["text"],
                metadata=res["entity"]["metadata"],
            )
            results.append(SearchResult(chunk=retrieved_chunk, similarity=res["distance"]))

        return results

    def get_stats(self):
        """Obtiene estadísticas de la colección"""
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            stats["row_count"] = self.client.query(
                collection_name=self.collection_name, filter="", output_fields=["count(*)"]
            )[0]["count(*)"]
            return stats
        except Exception as e:
            print(f"Error obteniendo estadísticas: {e}")
            return {"error": str(e)}
