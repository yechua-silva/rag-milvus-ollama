from pymilvus import MilvusClient
from typing import List, Dict
from tqdm import tqdm
from src.application.interfaces import VectorStore, Retriever


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

    def insert(self, data: List[Dict], batch_size: int = 100):
        """Insertar chunks en Milvus con embeddings de manera eficiente"""
        print(f"Insertando {len(data)} chunks en Milvus...")

        # Insertar en lotes
        for i in tqdm(range(0, len(data), batch_size), desc="Insertando lotes en Milvus"):
            batch = data[i : i + batch_size]
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

        print("Datos insertados")

        try:
            self.client.compact(collection_name=self.collection_name)
            print("Datos persistidos con compact")
        except Exception as e:
            print(f"Error haciendo compact: {e}")

    def search(self, vector: List[float], top_k: int) -> List[Dict]:
        """Busca en la base de conocimiento"""
        search_res = self.client.search(
            collection_name=self.collection_name, data=[vector], limit=top_k, output_fields=["text", "page", "source"]
        )

        results = [
            {
                "text": res["entity"]["text"],
                "metadata": {"page": res["entity"]["page"], "source": res["entity"]["source"]},
                "similarity": res["distance"],
            }
            for res in search_res[0]
        ]
        return results

    def get_stats(self):
        """Obtiene estadísticas de la colección"""
        try:
            stats = self.client.get_collection_stats(self.collection_name)

            try:
                result = self.client.search(
                    collection_name=self.collection_name,
                    data=[[0.0] * self.embedding_dim],
                    limit=10000,  # Límite alto
                    output_fields=["id"],
                )
                if result:
                    stats["row_count"] = len(result[0])
                else:
                    stats["row_count"] = 0
            except Exception as e:
                print(f"Error al obtener conteo: {e}")
                stats["row_count"] = "Desconocido"

            return stats
        except Exception as e:
            print(f"Error obteniendo estadísticas: {e}")
            return {"error": str(e)}
