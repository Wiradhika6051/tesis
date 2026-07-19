from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class Sample:

    # ========= Dataset =========

    repo: str
    commit_hash: str
    file_path: str

    source: str
    diff: str

    label: int

    # ========= CFG =========

    cfg: Optional[Any] = None
    line_to_node: Dict[int, int] = field(default_factory=dict)

    # ========= Localization =========

    seed_nodes: List[int] = field(default_factory=list)

    # ========= Pruning =========

    kept_nodes: List[int] = field(default_factory=list)
    pruned_cfg: Optional[Any] = None

    # ========= Model Input =========

    tokens: List[str] = field(default_factory=list)
    edges: List[Any] = field(default_factory=list)
    node_features: Optional[Any] = None

    # ========= Statistics =========

    statistics: Dict[str, Any] = field(default_factory=dict)

    