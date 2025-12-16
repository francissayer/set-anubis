from SetAnubis.core.MadGraph.adapters.input.MadGraphHepmcAnalyzer import MadGraphHepmcAnalyzer
import os

HEPMC_FILE = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "..", "Assets", "Test", "MadGraphOutput", "Events", "run_01_decayed_1", "tag_1_pythia8_events.hepmc.gz"))

if __name__ == "__main__":
    
    analyzer = MadGraphHepmcAnalyzer.from_file(HEPMC_FILE)

    stats = analyzer.analyze(
        pdg_id=35,
        max_events=None,  
        status=None, 
        ignore_self_decays=True, 
    )

    print(stats.summary())

    analyzer.plot_all(stats, bins=60, top_n_relations=15)