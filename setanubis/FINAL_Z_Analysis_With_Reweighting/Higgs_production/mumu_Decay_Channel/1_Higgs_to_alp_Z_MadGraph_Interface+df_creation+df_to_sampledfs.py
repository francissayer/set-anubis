from SetAnubis.core.MadGraph.adapters.input.GeneralCardInterface import GeneralCardInterface, MadGraphCommandConfig
from SetAnubis.core.MadGraph.adapters.input.MadGraphInterface import MadgraphInterface
from SetAnubis.core.interfaces import SetAnubisInterface

from SetAnubis.core.MadGraph.adapters.output.MadGraphLocalRunner import MadGraphLocalRunner

import os
import pyhepmc

from SetAnubis.core.ModelCore.adapters.input.SetAnubisInteface import SetAnubisInterface
from SetAnubis.core.Selection.domain.HepMCFrameBuilder import HepmcFrameBuilder, HepmcFrameOptions

from SetAnubis.core.Selection.domain.LLPAnalyzer import LLPAnalyzer
import pandas as pd

from SetAnubis.core.Selection.adapters.output.WriteLoadSelectionDict import save_bundle

if __name__ == "__main__":
    
    for gen in range(9,16):

        """
        Paramater use to choose if we want to produce the card only (True) or run madgraph on docker.
        """
        dry_run = False # True just checks that the script runs (i.e. just prints the cards, no runs), set to False to actually generate events
        
        """
        General interface of the neo-set-anubis pipeline. Need the path to the UFO as an input.
        
        Everything concerning this interface is available in the ModelCore.exampleNeoSetAnubisInterface.py example.
        """
        neo = SetAnubisInterface("/usera/fs568/set-anubis/Assets/UFO/ALP_linear_UFO_WIDTH_modified")
        
        """
        Configuration for the MagraphInterface (for writing cards). Few inputs are needed :
        
            -   neo_set_anubis : General NeoSetAnubis interface, with the ufo_path and all the particles/parameters.
            -   cards_path  :   Path to the cards in the docker container. No need to change it it will only break things (or be sure of what you're doing !).
            -   cache : If we want to use what's already in MadGraph, Generally put it to False only you're sure or doing the same scan than before.
            -   model_in_madgraph : name of the UFO, used in madgraph to import the model (from Feynrule).
            -   shower  :   shower option in madgraph, tell it which software will deal with the shower (use pythia by default).
            -   madspin :   madspin option, whether to use it for the decay of the LLP or not.
        """
        config = MadGraphCommandConfig(
            neo_set_anubis=neo,
            cache=False,
            model_in_madgraph="/usera/fs568/set-anubis/Assets/UFO/ALP_linear_UFO_WIDTH_modified",
            shower="py8",
            madspin="ON")
        
        """
        General interface for the cards creation. Everything on the heap, no file writing or anything.
        
        The three main cards are the runcard, param_card and the jobcard.
        
        param_card : The param_card is automatically created by the UFO (writing part) and will use the default value of the parameters. In order to change a parameters value, either change it in the UFO or 
        in the jobscard (parameter scan can be used with one value to set the parameters's value).
        
        run_card : The run_card can be edited to change the number of events, the parton distribution function, the energy of the beam, some cuts or other general parameters.
        
        jobcard : The jobcard is used to select the differents process to generate the LLP, and choose the parameters for the scan. See below the example.
        
        Two other cards are used for madspin and pythia : 
        
        pythia_card : The pythia_card is automatically generated and shouldn't be changed
        """
        card_interface = GeneralCardInterface(config)
        
        param_card = card_interface.param_card

        runcard_editor = card_interface.run_card_builder
        runcard_editor.set("nevents", 2000)
        
        #############################################################################################
        decay_channel_index = 2 # Unique index for the mumu decay channel, can be any integer (just used to differentiate the seed between different decay channels)
        process_index = 1 # Unique index for the pp > h > ax Z process, can be any integer (just used to differentiate the seed between different processes if needed)
        
        # Deterministic seed derived from generated-events (gen), decay channel number and process index
        # Compose seed = gen*1_000_000 + process_index*10_000 + decay_channel_index*1_000, fit to MadGraph's signed-32 limit (with modulo operation) and avoid 0
        # Generate a unique, reproducible Monte‑Carlo seed for this job.
        # We place the `gen` identifier in the millions place so different
        # Generated_Events groups occupy distinct ranges, and then rely on
        # MadGraph's internal behaviour to make each run unique within a scan.
        # MadGraph calls `update_random()` twice during the run flow which
        # advances its internal seed by +3 each call (effective +6 between
        # successive runs). So every run in a parameter scan ends up with a 
        # different seed (+6 to the seed for each run in the scan) while keeping the
        # overall value reproducible and easy to trace back to `gen`, the
        # decay channel index and the process index.
        #
        # example: for gen=1, decay_channel_index=1, process_index=1, seed = 1_000_000 + 10_000 + 1_000 = 1_011_000; then the runs in the scan will have seeds 1_011_000, 1_011_006, 1_011_012, etc.
        seed = (gen * 1_000_000 + process_index * 10_000 + decay_channel_index * 1_000) % 2147483647
        if seed == 0:
            seed = 1
        runcard_editor.set("iseed", seed)
        #############################################################################################
        
        #############################################################################################
        # # I've added this to use LHAPDF for PDFs in MadGraph (change ID to desired set)
        # runcard_editor.set("pdlabel", "lhapdf")
        # runcard_editor.set("lhaid", 331100)  # example: replace 306000 with your LHAPDF numeric ID
        #############################################################################################
        runcard_str = runcard_editor.serialize()

        builder_madspin = card_interface.madspin_builder
        # Include all possible final states - but don't include kinematically forbidden decays as MadSpin will error out
        #builder_madspin.add_decay("decay ax > e- e+")       # 2m_e = 1.022e-3 GeV
        builder_madspin.add_decay("decay ax > mu- mu+")     # 2m_mu = 0.21132 GeV
        #builder_madspin.add_decay("decay ax > ta- ta+")     # 2m_tau = 3.554 GeV
        #builder_madspin.add_decay("decay ax > u u~")        # 2m_u = 5.1e-3 GeV
        #builder_madspin.add_decay("decay ax > d d~")        # 2m_d = 1.008e-2 GeV
        #builder_madspin.add_decay("decay ax > s s~")        # 2m_s = 0.202 GeV
        #builder_madspin.add_decay("decay ax > c c~")        # 2m_c = 2.54 GeV
        #builder_madspin.add_decay("decay ax > b b~")        # 2m_b = 9.4 GeV
        #builder_madspin.add_decay("decay ax > t t~")        # 2m_t = 346.6 GeV
        #builder_madspin.add_decay("decay ax > ll ll")
        #builder_madspin.add_decay("decay ax > qq qq") 
        madspin_str = builder_madspin.serialize()
        
        pythia_str = card_interface.pythia_builder.serialize()
        #############################################################################################
        # # I've added this to ensure Pythia uses the same LHAPDF set (prepend these lines if not present)
        # pythia_pdf_header = "PDF:useLHAPDF = on\nPDF:LHAPDFset = NNPDF40_nnlo_as_01180\n"
        # if 'PDF:useLHAPDF' not in pythia_str:
        #     pythia_str = pythia_pdf_header + pythia_str
        #############################################################################################

        
        jobcard = card_interface.josbscript_builder
        #jobcard.add_process("define ll = e- e+ mu- mu+ ta- ta+ \n define qq = u u~ d d~ s s~ c c~ b b~ t t~ \n generate p p > h > ax Z")
        jobcard.add_process("generate p p > h > ax Z")
        jobcard.set_output_launch(f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/Higgs_to_ALP_Z_FINAL_Default_Lifetime_With_Reweighting/mumu_Decay_Channel/Generated_Events_{gen}/ALP_axZ_scan_1")
        jobcard.configure_cards()
        jobcard.add_auto_width("WALP")  # Automatically compute ALP width from decay formulas
        jobcard.add_parameter_scan("Ma", "[0.316,0.562,1,1.78,3.16,5.62,10,17.8,31.6]")      # ALP mass in GeV
        jobcard.add_parameter_scan("fa", "[1000]")     # ALP decay constant in GeV
        jobcard.add_parameter_scan("CaPhi", "[0.0001]")   # Universal ALP-fermion coupling
        jobcard.add_parameter_scan("CGtil", "[0.0]")   # ALP-Gluon coupling
        jobcard.add_parameter_scan("CWtil", "[0.0]")   # ALP-W coupling
        jobcard.add_parameter_scan("CBtil", "[0.0]")   # ALP-B coupling
        # Want just ALP-fermion coupling in this example
        jobscript_str = jobcard.serialize()

        print("------------------------------------------------------------------------------------------")
        print(jobscript_str)
        print("------------------------------------------------------------------------------------------")
        print(madspin_str)
        print("------------------------------------------------------------------------------------------")
        print(pythia_str)
        print("------------------------------------------------------------------------------------------")
        
        print(runcard_str)
        print("------------------------------------------------------------------------------------------")
        
        print(param_card)
        print("------------------------------------------------------------------------------------------")
        
        mlr = MadGraphLocalRunner(jobID=4) # Unique jobID to create a unique directory in MadGraph to store the temporary cards for this run and avoid conflicts with other runs. Can be any integer, just make sure it's different from the one used for other decay channels and processes to avoid conflicts in the MadGraph cache.
        
        if not dry_run:
            mg = MadgraphInterface(
                madgraph_runner=mlr,
                jobscript_str=jobscript_str,
                param_card_str=param_card,
                run_card_str=runcard_str,
                pythia_card_str=pythia_str,
                madspin_card_str=madspin_str
            )

            mg.run()
            mg.retrieve_events()
    

        
        
        
        
        
        
        
        
        
        
        ## DF creation from df_creation.py + DF to sampledfs from df_to_sampledfs.py
        
        UFO_PATH = os.path.abspath("/usera/fs568/set-anubis/Assets/UFO/ALP_linear_UFO_WIDTH_modified")

        neo = SetAnubisInterface(UFO_PATH)

        def on_progress(n: int):
            print(f"[build] {n} events")

        builder = HepmcFrameBuilder(
            neo_manager=neo,
            options=HepmcFrameOptions(progress_every=200, compute_met=True),
            progress_hook=on_progress,
        )
        
        for run in range(1,10):
            HEPMC_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/Higgs_to_ALP_Z_FINAL_Default_Lifetime_With_Reweighting/mumu_Decay_Channel/Generated_Events_{gen}/ALP_axZ_scan_1/Events/run_0{run}_decayed_1/tag_1_pythia8_events.hepmc.gz")

            with pyhepmc.open(HEPMC_FILE) as stream:
                df, unknown = builder.build_from_events(stream)
                
                df.to_pickle(f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/Higgs_to_ALP_Z_FINAL_Default_Lifetime_With_Reweighting/mumu_Decay_Channel/Generated_Events_{gen}/ALP_Z_df_Scan_1_Run_{run}.pkl")
            
            DF_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/Higgs_to_ALP_Z_FINAL_Default_Lifetime_With_Reweighting/mumu_Decay_Channel/Generated_Events_{gen}/ALP_Z_df_Scan_1_Run_{run}.pkl")

            df = pd.read_pickle(DF_FILE)
            LLPid = 9000005
            minPt = {"chargedTrack": 0.5}

            analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
            out_opt = analyzer.create_sample_dataframes(LLPid)
            
            save_bundle(out_opt, f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/Higgs_to_ALP_Z_FINAL_Default_Lifetime_With_Reweighting/mumu_Decay_Channel/Generated_Events_{gen}/ALP_Z_sampledfs_Scan_1_Run_{run}.pkl.gz")     # Want to save the sample dataframes for use in selection pipeline
            
            print(out_opt["LLPs"])