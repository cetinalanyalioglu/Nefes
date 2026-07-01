"""UI-provenance carrier: the layout/identity metadata a loaded case keeps.

``load_case`` reads only the physics out of a UI export (gas, elements, edges),
but the file also carries UI-only information Nefes does not model: canvas node
positions, id-generation counters, the human title, and the original node/edge
ids and port handles.  When the user saves the case back for the UI, that
information should round-trip verbatim so the network reopens exactly as drawn.

A :class:`UIProvenance` snapshots the parsed document plus the id ordering Nefes
assigned, and is stashed on the ``Network`` at load time (``net.provenance``).
The writer (:mod:`nefes.io.yaml_out`) reuses it for ids, port handles, positions,
counters and title, while refreshing the physical parameters from the live
``Network`` so any edits made in Python are reflected.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class UIProvenance:
    """UI-only metadata retained from a loaded case, indexed by Nefes node/edge id.

    Attributes
    ----------
    doc : dict
        The full parsed UI save document (``version``/``meta``/``model``/
        ``uiAttributes``/``uiState``), kept for verbatim re-emission of the
        layout and identity sections.
    node_ids : list of str
        Original UI node ids, ordered by Nefes node index (``node_ids[i]`` is the
        id of the element Nefes stored at index ``i``).
    edge_ids : list of str
        Original UI edge ids, ordered by Nefes edge index.
    """

    doc: dict
    node_ids: List[str] = field(default_factory=list)
    edge_ids: List[str] = field(default_factory=list)
