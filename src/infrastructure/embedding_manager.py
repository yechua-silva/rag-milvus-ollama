# src/infrastructure/embedding_manager.py
import ollama
from typing import List
from tqdm import tqdm
import time


class OllamaEmbeddingManager:
    """Sabe cómo generar embeddings usando Ollama."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        # Detectar la dimensión del embedding una vez
        self.embedding_dim = len(self.get_embedding("test"))

    def get_embedding(self, text: str):
        """Genera embeddings para el texto usando ollama"""
        # Limitar la longitud del texto para evitar problemas
        truncated_text = text[:4000]  # Limitar a 4000 caracteres
        max_retries = 3
        retry_delay = 1  # segundos

        for attempt in range(max_retries):
            try:
                response = ollama.embeddings(model=self.model_name, prompt=truncated_text)
                return response["embedding"]
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Intento {attempt + 1} fallido, reintentando en {retry_delay} segundos...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Backoff exponencial
                else:
                    print(f"Error generando embedding después de {max_retries} intentos: {e}")
                    # Devolver un embedding de ceros como fallback
                    return [0.0] * self.embedding_dim

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 15) -> List[List[float]]:
        """Genera embeddings para múltiples textos de manera eficiente"""
        embeddings = []

        # Procesar en lotes con delays para no saturar Ollama
        for i in tqdm(range(0, len(texts), batch_size), desc="Generando embeddings por lotes"):
            batch = texts[i : i + batch_size]
            batch_embeddings = []

            # Procesar cada texto en el lote
            for text in batch:
                embedding = self.get_embedding(text)
                batch_embeddings.append(embedding)

            embeddings.extend(batch_embeddings)

            # Pequeña pausa entre lotes para no sobrecargar Ollama
            time.sleep(0.3)

        return embeddings
