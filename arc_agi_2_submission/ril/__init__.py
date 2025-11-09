# RIL (Rule Induction Layer) package

from __future__ import annotations

from pathlib import Path
from typing import List

_this_dir = Path(__file__).resolve().parent
_repo_root = _this_dir.parents[2]
_arc_dir = _repo_root / "arc-agi-2-entry-2" / "ril"

__path__: List[str] = [str(_this_dir)]
if _arc_dir.is_dir():
    arc_path = str(_arc_dir)
    if arc_path not in __path__:
        __path__.insert(0, arc_path)
