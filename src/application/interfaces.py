from abc import ABC, abstractmethod
from typing import List, Dict, Any


class DocumentLoader(ABC):
    """Interface para cargar documentos desde diferentes fuentes"""

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """Carga documentos y devuelve una lista de páginas con metadatos"""
        pass


class TextProcessor(ABC):
    """Interface para procesar y limpiar texto"""

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto"""
        pass


class Chunker(ABC):
    """Interface para dividir texto en chunks"""

    @abstractmethod
    def chunk(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Divide las páginas en chunks preservando el contexto"""
        pass


class Embedder(ABC):
    """Interface para generar embeddings de texto"""

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Genera un embedding para un texto dado"""
        pass

    @abstractmethod
    def get_embeddings_batch(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Genera embeddings para múltiples textos de manera eficiente"""
        pass


class VectorStore(ABC):
    """Interface para almacenar y gestionar vectores"""

    @abstractmethod
    def set_collection(self):
        """Configura la colección en la base de datos vectorial"""
        pass

    @abstractmethod
    def insert(self, data: List[Dict], batch_size: int = 100):
        """Inserta datos en la base de datos vectorial"""
        pass


class Retriever(ABC):
    """Interface para recuperar información relevante"""

    @abstractmethod
    def search(self, vector: List[float], top_k: int) -> List[Dict]:
        """Busca los chunks más relevantes para un vector dado"""
        pass


class ResponseGenerator(ABC):
    """Interface para generar respuestas basadas en contexto"""

    @abstractmethod
    def generate_response(self, question: str, context: List[Dict]) -> str:
        """Genera una respuesta basada en la pregunta y el contexto proporcionado"""
        pass
