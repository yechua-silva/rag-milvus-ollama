import re
from typing import List
from tqdm import tqdm
from src.application.interfaces import TextProcessor, Chunker
from src.domain.models import DocumentPage, DocumentChunk


class SmartChunker(Chunker):
    """Sabe como limpiar y dividir texto en chunks inteligentes"""

    def __init__(self, text_processor: TextProcessor, chunk_size: int = 1000, overlap: int = 200):
        """Inicializa el chunker con configuración específica.

        Args:
            text_processor (TextProcessor): Procesador de texto para limpieza
            chunk_size (int, optional): Tamaño máximo de cada chunk en caracteres. Defaults to 1000.
            overlap (int, optional): Cantidad de caracteres que se superponen entre chunks. Defaults to 200.
        """
        self.text_processor = text_processor
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, pages_data: List[DocumentPage]) -> List[DocumentChunk]:
        """Divide en chunk preservando el contexto

        Args:
            pages_data (List[Dict]): Lista de paginas con texto

        Returns:
            List[Dict]: Lista de chunks con metadatos
        """
        chunks = []

        for page in tqdm(pages_data, desc="Creando chunks"):
            text = self.text_processor.clean_text(page.text)

            metadata = {"page": page.page_num, "source": page.source}

            if len(text) <= self.chunk_size:
                chunks.append(
                    DocumentChunk(doc_id=page.doc_id, text=text, metadata={**metadata, "chunk_type": "full_page"})
                )
            else:
                start = 0
                while start < len(text):
                    end = start + self.chunk_size
                    if end < len(text):
                        last_period = text.rfind(".", start, end)
                        if last_period > start + self.chunk_size // 2:
                            end = last_period + 1
                    chunk_text = text[start:end].strip()
                    if chunk_text:
                        chunk_metadata = {
                            **metadata,
                            "chunk_type": "partial_page",
                            "start_char": start,
                            "end_char": end,
                        }
                        chunks.append(DocumentChunk(doc_id=page.doc_id, text=chunk_text, metadata=chunk_metadata))
                    start += self.chunk_size - self.overlap
        return chunks


class BasicTextProcessor(TextProcessor):
    """Implementación concreta de TextProcessor para limpieza básica de texto"""

    def clean_text(self, text: str) -> str:
        """Limpia el texto extraido del PDF

        Args:
            text (str): Texto a limpiar

        Returns:
            str: Text limpio
        """
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\n", " ").replace("\r", " ")
        return text.strip()
