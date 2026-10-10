"""PSU PS2 mesh simplifier: QEM decimation + stripification + PS2 byte budgeting."""
from .mesh import Mesh
from .simplify import simplify, SimplifyOptions
from .strips import stripify, strips_to_faces
from .budget import estimate_bytes, ByteModel, fit_to_budget
__all__ = ["Mesh", "simplify", "SimplifyOptions", "stripify", "strips_to_faces",
           "estimate_bytes", "ByteModel", "fit_to_budget"]
