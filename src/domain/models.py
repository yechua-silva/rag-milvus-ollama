# src/domain/models.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from uuid import uuid4


@dataclass
class DocumentPage:
    page_num: int
    text: str
    source: str
    doc_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class DocumentChunk:
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    chunk: DocumentChunk
    similarity: float


@dataclass
class LLMResponse:
    answer: str
    source_chunks: List[SearchResult]
