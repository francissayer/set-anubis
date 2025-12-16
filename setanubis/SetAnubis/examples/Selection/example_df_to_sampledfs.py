from SetAnubis.core.Selection.domain.LLPAnalyzer import LLPAnalyzer
import pandas as pd
import os

DF_FILE = (os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "..", "Assets", "Test", "df_HNL_example", "perfect_df.csv")))

if __name__ == "__main__":
    df = pd.read_csv(DF_FILE)
    LLPid = 9900012
    minPt = {"chargedTrack": 0.5}

    analyzer = LLPAnalyzer(df.copy(), pt_min_cfg=minPt)
    out_opt = analyzer.create_sample_dataframes(LLPid)
    
    print(out_opt["LLPs"])
