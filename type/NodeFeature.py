from dataclasses import dataclass
from typing import List


@dataclass
class NodeFeature:

    node_id: int

    node_type: str

    text: str

    lineno: int