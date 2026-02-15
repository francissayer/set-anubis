"""
Width_Calculator.py - Direct analytical formulas from UFO

This script calculates the partial decay widths of Axion-Like Particles (ALPs) using the partial width formulae provided in the UFO model files in the decays.py script.

The decay formulae are associated with the particle objects defined in the UFO model (e.g. P.a for photon, P.g for gluon, etc.) and are stored in the Decay_ax.partial_widths dictionary in decays.py.

The input parameter for the particles are given by their PDG codes as integers (e.g. 22 for photon, 21 for gluon, etc.) defined in the UFO script particles.py and so are mapped to the corresponding UFO particle objects in this script.
"""
import sys
import os
from typing import Dict, Any, Set
from SetAnubis.core.BranchingRatio.domain.IDecayCalculation import IDecayCalculation

# Add UFO path to sys.path to allow imports of UFO modules
UFO_PATH = os.path.join(os.path.dirname(__file__), '..', 'Assets', 'UFO', 'ALP_linear_UFO_WIDTH')
if UFO_PATH not in sys.path:
    sys.path.insert(0, UFO_PATH)

# Now import from the UFO directory
from object_library import all_particles
from decays import Decay_ax

# The decay formulae use cmath for complex math operations
import cmath

class MyPythonDecayCalc(IDecayCalculation):

    def calculate(self, 
                  mother: int, 
                  daughters: Set[int], 
                  parameters: Dict[str, float]) -> float:
        
        # Only handle ALP decays (PDG code 9000005)
        if mother != 9000005:
            return 0.0
            
        # Create mapping from the daughter particle PDG codes given (integer) to particle object (from UFO to be inputted into Decay_ax.partial_widths to determine which partial width decay formula to use)
        pdg_to_particle = {p.pdg_code: p for p in all_particles}
        
        daughter_particles = tuple(pdg_to_particle[d] for d in daughters)
        
        # Next since the parameters dictionary is pulled from the SetAnubisInterface, we need to convert it to a dictionary format with name:value pairs for use in eval
        # Currently the parameters dictionary has the format name: { "value": complex number, ...} so we need to extract the "value" field
        numeric_parameters = {}
        for name, param in parameters.items():
            if isinstance(param, dict) and "value" in param:
                numeric_parameters[name] = param["value"]
            else:
                numeric_parameters[name] = param

        return eval(Decay_ax.partial_widths[daughter_particles], numeric_parameters, {'cmath': cmath})
        
        
        