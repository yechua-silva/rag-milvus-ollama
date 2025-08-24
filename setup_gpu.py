#!/usr/bin/env python3
"""
Script de configuración para GPU AMD con DirectML.

Este script automatiza la instalación de dependencias y preparación
de modelos necesarios para la aceleración GPU en el sistema RAG.
"""


def setup_models():
    """
    Prepara los modelos de embeddings necesarios.

    Descarga y configura el modelo de embeddings para su uso con ONNX
    y DirectML, mostrando información sobre la dimensión de embeddings.
    """
    from src.infrastructure.embedding_gpu import GPUEmbeddingGenerator

    print("Preparando modelo de embeddings...")
    embedder = GPUEmbeddingGenerator()
    print(f"Modelo listo. Dimensión de embeddings: {embedder.get_embedding_dim()}")


if __name__ == "__main__":
    """
    Punto de entrada principal del script de configuración.
    """
    print("Configurando entorno GPU...")
    setup_models()
    print("Configuración completada.")
