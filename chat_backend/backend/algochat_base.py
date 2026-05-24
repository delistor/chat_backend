"""
AlgoChat — Algorithm Base Module
Convention-based registration via @algorithm decorator + auto-discovery
"""

import os, importlib, inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Global registry
ALGORITHMS: Dict[str, "AlgorithmDef"] = {}


@dataclass
class Param:
    """Algorithm parameter definition."""
    key: str
    type: str = "int"            # int | float | select | text | bool
    default: Any = None
    label: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None   # for select type
    desc: str = ""

    def to_dict(self) -> dict:
        d = {"key": self.key, "type": self.type, "default": self.default,
             "label": self.label or self.key, "desc": self.desc}
        if self.min is not None: d["min"] = self.min
        if self.max is not None: d["max"] = self.max
        if self.step is not None: d["step"] = self.step
        if self.options is not None: d["options"] = self.options
        return d


@dataclass
class Input:
    """Algorithm input definition."""
    key: str
    types: List[str] = field(default_factory=lambda: ["csv", "xlsx", "json"])
    required: bool = True
    multiple: bool = False      # accept multiple files?
    desc: str = ""

    def to_dict(self) -> dict:
        return {"key": self.key, "types": self.types, "required": self.required,
                "multiple": self.multiple, "desc": self.desc or self.key}


@dataclass
class Output:
    """Algorithm output definition."""
    key: str
    type: str = "table"         # chart | table | image | document
    desc: str = ""
    group: str = ""             # optional group name for grouped display

    def to_dict(self) -> dict:
        d = {"key": self.key, "type": self.type, "desc": self.desc or self.key}
        if self.group:
            d["group"] = self.group
        return d


@dataclass
class AlgorithmDef:
    """Full algorithm definition stored in registry."""
    id: str
    name: str
    category: str
    icon: str
    desc: str = ""
    inputs: List[Input] = field(default_factory=list)
    outputs: List[Output] = field(default_factory=list)
    params: List[Param] = field(default_factory=list)
    handler: Callable = None

    def to_dict(self) -> dict:
        """Serialize for API (excludes handler)."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "icon": self.icon,
            "desc": self.desc,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "params": [p.to_dict() for p in self.params],
        }


def algorithm(
    id: str,
    name: str,
    category: str,
    icon: str = "⚙",
    desc: str = "",
    inputs: List[Input] = None,
    outputs: List[Output] = None,
    params: List[Param] = None,
):
    """Decorator to register an algorithm function.

    The decorated function receives (inputs: dict, params: dict) and must
    return a dict mapping output keys to their values.

    For 'chart' outputs, value = {"chartType": "...", "datasets": [...]} or {"labels": [...], "datasets": [...]}
    For 'table' outputs, value = {"columns": [...], "rows": [...]}
    For 'image' outputs, value = {"src": "/api/images/xxx.png", "name": "..."}
    For 'document' outputs, value = str (plain text / markdown)
    """
    def decorator(func: Callable):
        algo = AlgorithmDef(
            id=id, name=name, category=category, icon=icon, desc=desc,
            inputs=inputs or [], outputs=outputs or [], params=params or [],
            handler=func,
        )
        ALGORITHMS[id] = algo
        func._algo_def = algo
        return func
    return decorator


def discover_algorithms(directory: str = None):
    """Auto-discover and import all algorithm modules in a directory.

    Scans all .py files (excluding __init__.py and private _*.py) in the
    given directory and imports them. Any function decorated with @algorithm
    will be automatically registered.
    """
    if directory is None:
        directory = os.path.join(os.path.dirname(__file__), "algorithms")

    if not os.path.isdir(directory):
        print(f"⚠ Algorithms directory not found: {directory}")
        return

    import sys
    # Ensure parent package is importable
    parent = os.path.dirname(directory)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    package_name = os.path.basename(directory)
    count_before = len(ALGORITHMS)

    for filename in sorted(os.listdir(directory)):
        if filename.startswith("_") or not filename.endswith(".py"):
            continue
        module_name = filename[:-3]
        full_module = f"{package_name}.{module_name}"
        try:
            importlib.import_module(full_module)
        except Exception as e:
            print(f"❌ Failed to load algorithm module {full_module}: {e}")

    count_after = len(ALGORITHMS)
    ids = ", ".join(ALGORITHMS.keys())
    print(f"✅ 已注册 {count_after} 个算法: {ids}")


def get_algorithm(algo_id: str) -> Optional[AlgorithmDef]:
    return ALGORITHMS.get(algo_id)


def list_algorithms() -> List[dict]:
    return [a.to_dict() for a in ALGORITHMS.values()]