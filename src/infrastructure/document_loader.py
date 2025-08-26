import fitz
import os
from typing import List
from glob import glob
from src.application.interfaces import DocumentLoader
from src.domain.models import DocumentPage


class PdfDocumentLoader(DocumentLoader):
    """Sabe como leer multiples archivos PDF y extraer su texto por pagina"""

    def __init__(self, docs_folder: str = "./docs"):
        self.docs_folder = docs_folder
        if not os.path.exists(docs_folder):
            os.makedirs(docs_folder)

    def load(self) -> List[DocumentPage]:
        """Extrae texto de todos los PDFs en la carpeta por pagina con metadatos"""
        pdf_files = glob(os.path.join(self.docs_folder, "*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No se encontraron archivos PDF en {self.docs_folder}")

        all_pages = []
        for pdf_path in pdf_files:
            print(f"Procesando: {os.path.basename(pdf_path)}")
            try:
                doc = fitz.open(pdf_path)
                doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        document_page = DocumentPage(
                            page_num=page_num + 1, text=text, source=os.path.basename(pdf_path), doc_id=doc_id
                        )
                        all_pages.append(document_page)
                doc.close()
            except Exception as e:
                print(f"Error procesando {pdf_path}: {e}")

        return all_pages
