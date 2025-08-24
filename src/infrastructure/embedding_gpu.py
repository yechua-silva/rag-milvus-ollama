import onnxruntime as ort
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import os
from tqdm import tqdm


class GPUEmbeddingGenerator:
    """
    Generador de embeddings optimizado para GPU mediante ONNX Runtime.

    Esta clase permite la generación eficiente de embeddings utilizando modelos
    de sentence transformers convertidos a formato ONNX, con soporte para
    aceleración GPU a través de DirectML (AMD) o CUDA (NVIDIA).

    Args:
        model_name (str): Nombre del modelo de embeddings a utilizar

    Attributes:
        model_name (str): Nombre del modelo de embeddings
        tokenizer: Tokenizer del modelo
        providers (list): Proveedores de ejecución disponibles
        session: Sesión de inferencia ONNX
        embedding_dim (int): Dimensión de los embeddings generados
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inicializa el generador de embeddings con GPU.

        Args:
            model_name (str): Nombre del modelo de embeddings. Defaults to "sentence-transformers/all-MiniLM-L6-v2"
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.providers = self._get_available_providers()
        self.session = self._load_onnx_model()
        self.embedding_dim = 384

    def get_embedding(self, text: str) -> List[float]:
        """Genera un embedding para un solo texto."""
        # Reutilizamos de forma eficiente la función de batch, pero con un solo elemento.
        embedding_array = self.generate_embeddings([text])
        return embedding_array[0].tolist()

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos en lotes (batches).
        Esta función ahora es compatible con el Orchestrator.
        """
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generando embeddings con GPU"):
            batch = texts[i : i + batch_size]
            if not batch:
                continue  # Evita procesar lotes vacíos
            batch_embeddings = self.generate_embeddings(batch)
            all_embeddings.extend(batch_embeddings.tolist())
        return all_embeddings

    def _get_available_providers(self):
        """
        Detecta y retorna los proveedores de ejecución disponibles.

        Returns:
            List[str]: Lista de proveedores disponibles, priorizando GPU
        """
        available_providers = ort.get_available_providers()
        providers = []
        if "DmlExecutionProvider" in available_providers:
            providers.append("DmlExecutionProvider")
            print("Usando DirectML para aceleración GPU AMD")
        elif "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
            print("Usando CUDA para aceleración GPU NVIDIA")
        else:
            providers.append("CPUExecutionProvider")
            print("Usando CPU - no se encontraron proveedores GPU")
        return providers

    def _load_onnx_model(self):
        """
        Carga el modelo ONNX desde disco o lo exporta si no existe.

        Returns:
            Sesión de inferencia ONNX configurada
        """
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{self.model_name.replace('/', '_')}.onnx"
        try:
            session = ort.InferenceSession(model_path, providers=self.providers)
            print(f"Modelo ONNX cargado desde {model_path}")
            return session
        except Exception:
            print("Modelo ONNX no encontrado, re-exportando desde HuggingFace...")
            return self._export_and_load_onnx(model_path)

    def _export_and_load_onnx(self, model_path: str):
        """
        Exporta el modelo HuggingFace a formato ONNX y lo carga.

        Args:
            model_path (str): Ruta donde guardar el modelo exportado
        """
        model = AutoModel.from_pretrained(self.model_name)
        dummy_input = self.tokenizer("dummy input", return_tensors="pt", padding=True, truncation=True, max_length=512)

        torch.onnx.export(
            model,
            tuple(dummy_input.values()),
            model_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
            opset_version=14,
        )
        print(f"Modelo exportado a {model_path}")
        return ort.InferenceSession(model_path, providers=self.providers)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings para una lista de textos.

        Args:
            texts (List[str]): Lista de textos a procesar

        Returns:
            np.ndarray: Array con los embeddings generados
        """
        if not texts:
            return np.array([])

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np", max_length=512)
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs["token_type_ids"],
        }
        outputs = self.session.run(None, ort_inputs)[0]
        embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
        return embeddings

    def _mean_pooling(self, model_output, attention_mask):
        """
        Aplica pooling promedio a las salidas del modelo.

        Args:
            model_output: Salida del modelo de transformers
            attention_mask: Máscara de atención

        Returns:
            np.ndarray: Embeddings con pooling aplicado
        """
        token_embeddings = model_output
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), 1e-9, None)
        return sum_embeddings / sum_mask

    def get_embedding_dim(self):
        """
        Retorna la dimensión de los embeddings generados.

        Returns:
            int: Dimensión de los embeddings
        """
        return self.embedding_dim
