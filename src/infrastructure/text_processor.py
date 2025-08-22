import re
from typing import List, Dict
from tqdm import tqdm


class SmartChunker:
    """Sabe como limpiar y dividir texto en chunks inteligentes"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """Inicializa el chunker con configuración específica.

        Args:
            chunk_size (int, optional): Tamaño maximo de cada chunk en caracteres. Defaults to 1000.
            overlap (int, optional): Cantidad de caracteres que se superponen entre chunks. Defaults to 200.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _clean_text(self, text: str) -> str:
        """Limpia el texto extraido del PDF

        Args:
            text (str): Texto a limpiar

        Returns:
            str: Text limpio
        """
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\n", " ").replace("\r", " ")
        return text.strip()

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
            text = self._clean_text(page_data["text"])  # Corregido: page_data en lugar de pages_data
            page_num = page_data["page"]  # Corregido: page_data en lugar de pages_data
            source = page_data["source"]  # Corregido: page_data en lugar de pages_data

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
