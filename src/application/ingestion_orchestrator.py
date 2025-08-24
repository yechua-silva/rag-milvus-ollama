import numpy as np
from multiprocessing import Pool, cpu_count
from typing import List
import logging
from tqdm import tqdm
import fitz
import re
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _process_batch_worker(batch_data):
    """
    Función que se ejecuta en cada proceso worker.
    Inicializa su propio embedder y procesa un batch.
    """
    from src.infrastructure.embedding_gpu import GPUEmbeddingGenerator

    batch_texts, model_name = batch_data
    # Cada worker crea su propia instancia, abriendo su propia conexión a la GPU.
    embedder = GPUEmbeddingGenerator(model_name=model_name)
    try:
        return embedder.generate_embeddings(batch_texts)
    except Exception as e:
        # Si un worker falla, devolvemos ceros para no romper todo el proceso.
        logger.error(f"Error procesando un batch en un worker: {e}")
        return np.zeros((len(batch_texts), embedder.get_embedding_dim()))


class IngestionOrchestrator:
    """
    Orquestador del proceso de ingesta de documentos con soporte para GPU.

    Coordina la extracción, chunking, generación de embeddings e inserción
    en la base de datos vectorial, utilizando procesamiento paralelo para
    optimizar el rendimiento.

    Args:
        milvus_store: Cliente de Milvus para almacenamiento vectorial
        docs_folder (str): Ruta a la carpeta con documentos PDF
        batch_size (int): Tamaño de lote para procesamiento. Defaults to 64
        num_workers (int): Número de workers para procesamiento paralelo
    """

    def __init__(self, milvus_store, docs_folder: str, config, num_workers: int = None):
        """
        Inicializa el orquestador de ingesta.
        """
        self.docs_folder = docs_folder
        self.milvus_store = milvus_store
        self.config = config
        self.num_workers = num_workers or max(1, cpu_count() - 1)

    def process_documents(self):
        """
        Procesa todos los documentos PDF en la carpeta configurada.

        Returns:
            int: Número total de chunks procesados

        Raises:
            FileNotFoundError: Si no se encuentran documentos PDF
        """
        pdf_files = [f for f in os.listdir(self.docs_folder) if f.endswith(".pdf")]
        if not pdf_files:
            raise FileNotFoundError(f"No se encontraron archivos PDF en {self.docs_folder}")

        total_chunks = 0
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.docs_folder, pdf_file)
            chunks_processed = self.process_document(pdf_path)
            total_chunks += chunks_processed
            print(f"Procesado {pdf_file} con {chunks_processed} chunks")

        print(f"Documentos procesados: {len(pdf_files)} \nChunks en total: {total_chunks}")
        return total_chunks

    def process_document(self, file_path: str):
        """
        Procesa un documento PDF individual.

        Args:
            file_path (str): Ruta al archivo PDF a procesar

        Returns:
            int: Número de chunks extraídos del documento

        Raises:
            Exception: Si ocurre algún error durante el procesamiento
        """

        try:
            chunks, metadata = self._extract_and_chunk(file_path)
            embeddings = self._generate_embeddings_parallel(chunks)

            data_to_insert = []
            chunk_id_counter = 0  # Usamos un contador local para los IDs
            for chunk_text, meta in zip(chunks, metadata):
                embedding_vector = embeddings[chunk_id_counter].tolist()
                data_to_insert.append(
                    {
                        "id": chunk_id_counter,
                        "vector": embedding_vector,
                        "text": chunk_text[:2000],  # Limitar tamaño de texto por si acaso
                        "page": meta["page"],
                        "source": meta["source"],
                    }
                )
                chunk_id_counter += 1

            self.milvus_store.insert(data_to_insert, batch_size=100)
            logger.info(f"Documento: {file_path} - Chunks: {len(chunks)}")
            return len(chunks)
        except Exception as e:
            logger.error(f"Error procesando documento {file_path}: {e}")
            raise

    def _extract_and_chunk(self, file_path: str):
        """
        Extrae texto de un PDF y lo divide en chunks.

        Args:
            file_path (str): Ruta al archivo PDF

        Returns:
            tuple: (chunks, metadata) - Listas de chunks y metadatos

        Raises:
            Exception: Si ocurre error en la extracción del PDF
        """
        chunks = []
        metadata = []
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                text = re.sub(r"\s+", " ", text).replace("\n", " ").strip()
                start = 0
                while start < len(text):
                    end = start + self.config.CHUNK_SIZE
                    chunk_text = text[start:end].strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                        metadata.append({"page": page_num + 1, "source": os.path.basename(file_path)})
                    start += self.config.CHUNK_SIZE - self.config.CHUNK_OVERLAP
        doc.close()
        return chunks, metadata

    def _generate_embeddings_parallel(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings en paralelo utilizando múltiples workers.

        Args:
            embedder: Instancia del generador de embeddings
            texts (List[str]): Lista de textos a procesar

        Returns:
            np.ndarray: Array con todos los embeddings generados
        """

        if not texts:
            return np.array([])

        batches = [
            texts[i : i + self.config.EMBEDDING_BATCH_SIZE]
            for i in range(0, len(texts), self.config.EMBEDDING_BATCH_SIZE)
        ]
        worker_data = [(batch, self.config.EMBEDDING_ONNX_MODEL) for batch in batches]

        with Pool(self.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_process_batch_worker, worker_data),
                    total=len(batches),
                    desc="Generando embeddings en paralelo",
                )
            )
        return np.vstack(results)

    def _process_batch(self, args):
        """
        Procesa un batch de textos (ejecutado en cada worker).

        Args:
            args: Tuple con (embedder, batch)

        Returns:
            np.ndarray: Embeddings del batch procesado
        """
        embedder, batch = args
        try:
            return embedder.generate_embeddings(batch)
        except Exception as e:
            logger.error(f"Error procesando batch: {e}")
            return np.zeros((len(batch), embedder.embedding_dim))
