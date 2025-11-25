from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class NewParticlePythiaConfig:
    id: int
    is_visible : bool = False
    is_resonance : bool = False
    tau_calc : bool = False
    may_decay : bool = True
    do_external_decays : bool = False
    do_force_width : bool = True