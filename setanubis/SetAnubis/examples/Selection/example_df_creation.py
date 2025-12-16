import os
import pyhepmc

from SetAnubis.core.ModelCore.adapters.input.SetAnubisInteface import SetAnubisInterface
from SetAnubis.core.Selection.domain.HepMCFrameBuilder import HepmcFrameBuilder, HepmcFrameOptions

HEPMC_FILE = (
    "db/Temp/madgraph/Events/Events/run_01_decayed_1/"
    "tag_1_pythia8_events.hepmc/tag_1_pythia8_events.hepmc"
)

HEPMC_FILE = (os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "..", "Assets", "Test", "MadGraphOutput", "Events", "run_01_decayed_1", "tag_1_pythia8_events.hepmc.gz")))


UFO_HAHM_DIR = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "..", "Assets", "UFO", "HAHM_variableMW_v5_UFO"))

if __name__ == "__main__":
    neo = SetAnubisInterface(UFO_HAHM_DIR)

    def on_progress(n: int):
        print(f"[build] {n} events")

    builder = HepmcFrameBuilder(
        neo_manager=neo,
        options=HepmcFrameOptions(progress_every=200, compute_met=False),
        progress_hook=on_progress,
    )

    with pyhepmc.open(HEPMC_FILE) as stream:
        df, unknown = builder.build_from_events(stream)
        
        df.to_pickle("perfect_df.pkl")