# src/type/Sample.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from torch_geometric.data import Data
from src.type.NodeFeature import NodeFeature


@dataclass
class Sample:
    # Dataset identity
    repo: str
    parent_commit: str
    commit_hash: str
    file_path: str
    source: str
    diff: str
    label: int

    # Revision / pairing identity
    fix_commit_hash: str = ""
    revision: str = ""
    function_pair_id: Optional[str] = None
    pair_role: Optional[str] = None

    # Function metadata
    function_name: Optional[str] = None
    qualified_name: Optional[str] = None
    function_start: Optional[int] = None
    function_end: Optional[int] = None
    function_source: Optional[str] = None
    function_is_async: bool = False
    parent_sample_id: Optional[str] = None
    sample_id: Optional[str] = None

    # CFG
    cfg: Optional[Any] = None
    line_to_node: Dict[int, int] = field(default_factory=dict)

    # Localization
    seed_lines: List[int] = field(default_factory=list)
    seed_nodes: List[int] = field(default_factory=list)

    # Pruning
    kept_nodes: List[int] = field(default_factory=list)
    pruned_cfg: Optional[Any] = None

    # Model input
    node_features: Optional[List[NodeFeature]] = None
    graph: Optional[Data] = None
    function_nodes: List[int] = field(default_factory=list)

    # Statistics
    statistics: Dict[str, Any] = field(default_factory=dict)