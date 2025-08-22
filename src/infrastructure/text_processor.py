import re
from typing import List, Dict
from tqdm import tqdm
from src.application.interfaces import TextProcessor, Chunker


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

    def chunk(self, pages_data: List[Dict]) -> List[Dict]:
        """Divide en chunk preservando el contexto

        Args:
            pages_data (List[Dict]): Lista de paginas con texto

        Returns:
            List[Dict]: Lista de chunks con metadatos
        """
        chunks = []
        chunk_id = 0

        for page_data in tqdm(pages_data, desc="Creando chunks..."):
            text = self.text_processor.clean_text(page_data["text"])
            page_num = page_data["page"]
            source = page_data["source"]

            # si la pagina es pequeña, usar toda la pagina
            if len(text) <= self.chunk_size:
                chunks.append(
                    {"id": chunk_id, "text": text, "page": page_num, "source": source, "chunk_type": "full_page"}
                )
                chunk_id += 1
            else:
                # dividir paginas largas en chunks
                start = 0
                while start < len(text):
                    end = start + self.chunk_size

                    # intentar cortar en punto natural
                    if end < len(text):
                        last_period = text.rfind(".", start, end)
                        if last_period > start + self.chunk_size // 2:
                            end = last_period + 1

                    chunk_text = text[start:end].strip()

                    if chunk_text:
                        chunks.append(
                            {
                                "id": chunk_id,
                                "text": chunk_text,
                                "page": page_num,
                                "source": source,
                                "chunk_type": "partial_page",
                                "start_char": start,
                                "end_char": end,
                            }
                        )
                        chunk_id += 1

                    # mover el inicio con overlap
                    start = end - self.overlap
                    if start >= len(text):
                        break
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
