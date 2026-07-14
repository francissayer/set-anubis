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
    
    #for i in range(1,13):
    for i in range(12,13):
        gen = 1
        scan = i

        
        
        
        
        
        
        
        
        
        
        ## DF creation from df_creation.py + DF to sampledfs from df_to_sampledfs.py
        
        UFO_PATH = os.path.abspath("/usera/fs568/set-anubis/Assets/UFO/ALP_linear_UFO_WIDTH")

        neo = SetAnubisInterface(UFO_PATH)

        def on_progress(n: int):
            print(f"[build] {n} events")

        builder = HepmcFrameBuilder(
            neo_manager=neo,
            options=HepmcFrameOptions(progress_every=200, compute_met=True),
            progress_hook=on_progress,
        )
        
        if i == 1:
            for run in range(1,7):
                HEPMC_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_1/ALP_axW+_scan_{scan}/Events/run_0{run}_decayed_1/tag_1_pythia8_events.hepmc.gz")

                with pyhepmc.open(HEPMC_FILE) as stream:
                    df, unknown = builder.build_from_events(stream)
                    
                    df.to_pickle(f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")
                
                DF_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                df = pd.read_pickle(DF_FILE)
                LLPid = 9000005
                minPt = {"chargedTrack": 0.5}

                analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
                out_opt = analyzer.create_sample_dataframes(LLPid)
                
                save_bundle(out_opt, f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_sampledfs_Scan_{scan}_Run_{run}.pkl.gz")     # Want to save the sample dataframes for use in selection pipeline
                
                print(out_opt["LLPs"])
        
        elif i == 2:
            for run in range(1,9):
                HEPMC_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_axW+_scan_{scan}/Events/run_0{run}_decayed_1/tag_1_pythia8_events.hepmc.gz")

                with pyhepmc.open(HEPMC_FILE) as stream:
                    df, unknown = builder.build_from_events(stream)
                    
                    df.to_pickle(f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")
        
                DF_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                df = pd.read_pickle(DF_FILE)
                LLPid = 9000005
                minPt = {"chargedTrack": 0.5}

                analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
                out_opt = analyzer.create_sample_dataframes(LLPid)
                
                save_bundle(out_opt, f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_sampledfs_Scan_{scan}_Run_{run}.pkl.gz")     # Want to save the sample dataframes for use in selection pipeline
                
                print(out_opt["LLPs"])        
        
        elif i == 3:
            for run in range(1,10):
                HEPMC_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_axW+_scan_{scan}/Events/run_0{run}_decayed_1/tag_1_pythia8_events.hepmc.gz")

                with pyhepmc.open(HEPMC_FILE) as stream:
                    df, unknown = builder.build_from_events(stream)
                    
                    df.to_pickle(f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")
        
                DF_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                df = pd.read_pickle(DF_FILE)
                LLPid = 9000005
                minPt = {"chargedTrack": 0.5}

                analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
                out_opt = analyzer.create_sample_dataframes(LLPid)
                
                save_bundle(out_opt, f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_sampledfs_Scan_{scan}_Run_{run}.pkl.gz")     # Want to save the sample dataframes for use in selection pipeline
                
                print(out_opt["LLPs"])        
        
        elif i == 4:
            for run in range(1,10):
                HEPMC_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_axW+_scan_{scan}/Events/run_0{run}_decayed_1/tag_1_pythia8_events.hepmc.gz")

                with pyhepmc.open(HEPMC_FILE) as stream:
                    df, unknown = builder.build_from_events(stream)
                    
                    df.to_pickle(f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")
        
                DF_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                df = pd.read_pickle(DF_FILE)
                LLPid = 9000005
                minPt = {"chargedTrack": 0.5}

                analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
                out_opt = analyzer.create_sample_dataframes(LLPid)
                
                save_bundle(out_opt, f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_sampledfs_Scan_{scan}_Run_{run}.pkl.gz")     # Want to save the sample dataframes for use in selection pipeline
                
                print(out_opt["LLPs"])        
        
        elif i == 5:
            for run in range(1,10):
                HEPMC_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_axW+_scan_{scan}/Events/run_0{run}_decayed_1/tag_1_pythia8_events.hepmc.gz")

                with pyhepmc.open(HEPMC_FILE) as stream:
                    df, unknown = builder.build_from_events(stream)
                    
                    df.to_pickle(f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                DF_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                df = pd.read_pickle(DF_FILE)
                LLPid = 9000005
                minPt = {"chargedTrack": 0.5}

                analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
                out_opt = analyzer.create_sample_dataframes(LLPid)
                
                save_bundle(out_opt, f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_sampledfs_Scan_{scan}_Run_{run}.pkl.gz")     # Want to save the sample dataframes for use in selection pipeline
                
                print(out_opt["LLPs"])

        elif i == 6:
            for run in range(1,9):
                HEPMC_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_axW+_scan_{scan}/Events/run_0{run}_decayed_1/tag_1_pythia8_events.hepmc.gz")

                with pyhepmc.open(HEPMC_FILE) as stream:
                    df, unknown = builder.build_from_events(stream)
                    
                    df.to_pickle(f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                DF_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                df = pd.read_pickle(DF_FILE)
                LLPid = 9000005
                minPt = {"chargedTrack": 0.5}

                analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
                out_opt = analyzer.create_sample_dataframes(LLPid)
                
                save_bundle(out_opt, f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_sampledfs_Scan_{scan}_Run_{run}.pkl.gz")     # Want to save the sample dataframes for use in selection pipeline
                
                print(out_opt["LLPs"])

        elif i == 7:
            for run in range(1,9):
                HEPMC_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_axW+_scan_{scan}/Events/run_0{run}_decayed_1/tag_1_pythia8_events.hepmc.gz")

                with pyhepmc.open(HEPMC_FILE) as stream:
                    df, unknown = builder.build_from_events(stream)
                    
                    df.to_pickle(f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                DF_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                df = pd.read_pickle(DF_FILE)
                LLPid = 9000005
                minPt = {"chargedTrack": 0.5}

                analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
                out_opt = analyzer.create_sample_dataframes(LLPid)
                
                save_bundle(out_opt, f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_sampledfs_Scan_{scan}_Run_{run}.pkl.gz")     # Want to save the sample dataframes for use in selection pipeline
                
                print(out_opt["LLPs"])

        elif i == 8:
            for run in range(1,9):
                HEPMC_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_axW+_scan_{scan}/Events/run_0{run}_decayed_1/tag_1_pythia8_events.hepmc.gz")

                with pyhepmc.open(HEPMC_FILE) as stream:
                    df, unknown = builder.build_from_events(stream)
                    
                    df.to_pickle(f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                DF_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                df = pd.read_pickle(DF_FILE)
                LLPid = 9000005
                minPt = {"chargedTrack": 0.5}

                analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
                out_opt = analyzer.create_sample_dataframes(LLPid)
                
                save_bundle(out_opt, f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_sampledfs_Scan_{scan}_Run_{run}.pkl.gz")     # Want to save the sample dataframes for use in selection pipeline
                
                print(out_opt["LLPs"])

        elif i == 9:
            for run in range(1,9):
                HEPMC_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_1/ALP_axW+_scan_{scan}/Events/run_0{run}_decayed_1/tag_1_pythia8_events.hepmc.gz")

                with pyhepmc.open(HEPMC_FILE) as stream:
                    df, unknown = builder.build_from_events(stream)
                    
                    df.to_pickle(f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                DF_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                df = pd.read_pickle(DF_FILE)
                LLPid = 9000005
                minPt = {"chargedTrack": 0.5}

                analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
                out_opt = analyzer.create_sample_dataframes(LLPid)
                
                save_bundle(out_opt, f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_sampledfs_Scan_{scan}_Run_{run}.pkl.gz")     # Want to save the sample dataframes for use in selection pipeline
                
                print(out_opt["LLPs"])

        elif i == 10:
            for run in range(1,9):
                HEPMC_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_1/ALP_axW+_scan_{scan}/Events/run_0{run}_decayed_1/tag_1_pythia8_events.hepmc.gz")

                with pyhepmc.open(HEPMC_FILE) as stream:
                    df, unknown = builder.build_from_events(stream)
                    
                    df.to_pickle(f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                DF_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                df = pd.read_pickle(DF_FILE)
                LLPid = 9000005
                minPt = {"chargedTrack": 0.5}

                analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
                out_opt = analyzer.create_sample_dataframes(LLPid)
                
                save_bundle(out_opt, f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_sampledfs_Scan_{scan}_Run_{run}.pkl.gz")     # Want to save the sample dataframes for use in selection pipeline
                
                print(out_opt["LLPs"])

        elif i == 11:
            for run in range(1,8):
                HEPMC_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_1/ALP_axW+_scan_{scan}/Events/run_0{run}_decayed_1/tag_1_pythia8_events.hepmc.gz")

                with pyhepmc.open(HEPMC_FILE) as stream:
                    df, unknown = builder.build_from_events(stream)
                    
                    df.to_pickle(f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                DF_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                df = pd.read_pickle(DF_FILE)
                LLPid = 9000005
                minPt = {"chargedTrack": 0.5}

                analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
                out_opt = analyzer.create_sample_dataframes(LLPid)
                
                save_bundle(out_opt, f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_sampledfs_Scan_{scan}_Run_{run}.pkl.gz")     # Want to save the sample dataframes for use in selection pipeline
                
                print(out_opt["LLPs"])

        elif i == 12:
            for run in range(1,7):
                HEPMC_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_axW+_scan_{scan}/Events/run_0{run}_decayed_1/tag_1_pythia8_events.hepmc.gz")

                with pyhepmc.open(HEPMC_FILE) as stream:
                    df, unknown = builder.build_from_events(stream)
                    
                    df.to_pickle(f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                DF_FILE = (f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_df_Scan_{scan}_Run_{run}.pkl")

                df = pd.read_pickle(DF_FILE)
                LLPid = 9000005
                minPt = {"chargedTrack": 0.5}

                analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
                out_opt = analyzer.create_sample_dataframes(LLPid)
                
                save_bundle(out_opt, f"/raid/anubis/sensitivityStudyData/ALPs/fermionCoupled/ALP_W+_Runs_2/Generated_Events_{gen}/ALP_W+_sampledfs_Scan_{scan}_Run_{run}.pkl.gz")     # Want to save the sample dataframes for use in selection pipeline
                
                print(out_opt["LLPs"])
        
    