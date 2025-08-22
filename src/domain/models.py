from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DocumentChunk:
    """Representa un trozo de documento"""

    id: int
    text: str
    metadata: Dict[str, Any]
