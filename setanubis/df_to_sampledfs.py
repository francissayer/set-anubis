from SetAnubis.core.Selection.domain.LLPAnalyzer import LLPAnalyzer
import pandas as pd
import os

from SetAnubis.core.Selection.adapters.output.WriteLoadSelectionDict import save_bundle

DF_FILE = ("/usera/fs568/set-anubis/ALP_W+_df_Scan_7_Run_2.pkl")

if __name__ == "__main__":
    df = pd.read_pickle(DF_FILE)
    LLPid = 9000005
    minPt = {"chargedTrack": 0.5}

    analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
    out_opt = analyzer.create_sample_dataframes(LLPid)
    
    save_bundle(out_opt, "ALP_W+_sampledfs_Scan_7_Run_2.pkl.gz")     # Want to save the sample dataframes for use in selection pipeline
    
    print(out_opt["LLPs"])
