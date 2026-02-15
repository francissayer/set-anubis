import os
import argparse
import matplotlib.pyplot as plt

from SetAnubis.core.Selection.domain.SelectionPipeline import SelectionPipelineBuilder
from SetAnubis.core.Selection.domain.SelectionManager import SelectionManager
from SetAnubis.core.Selection.domain.DatasetSource import EventsBundleSource
from SetAnubis.core.Selection.domain.SelectionEngine import SelectionConfig, RunConfig, MinThresholds, MinDR
from SetAnubis.core.Selection.adapters.input.SelectionGeometryAdapter import SelectionGeometryAdapter
from SetAnubis.core.Geometry.adapters.selection_adapter import GeometrySelectionAdapter
from SetAnubis.core.Geometry.domain.defineGeometry import ATLASCavern


def build_selection(sel_mode: str = "standard"):
    cav = ATLASCavern()
    geom_adapter = GeometrySelectionAdapter(cav)
    sel_geo = SelectionGeometryAdapter(geom_adapter)
    
    cav.createSimpleRPCs([cav.archRadius-0.2, cav.archRadius-0.6, cav.archRadius-1.2], RPCthickness=0.06)   # I added this line to create RPCs, otherwise intersectANUBISstationsSimple fails? This is consistent with ceiling configuration where in defineGeometry.py
                                                                                                            # line 1202 the simple RPCs are called ceiling (self.geoMode = "ceiling") and at then end in line 1365 in the __main__ section, this exact same call is in 
                                                                                                            # the if args.mode in ["", "simple"] block

    sel_cfg = SelectionConfig(
        geometry=sel_geo,
        minMET=30.0,
        minP=MinThresholds(LLP=0.1, chargedTrack=0.1, neutralTrack=0.1, jet=0.1),
        minPt=MinThresholds(LLP=0.0, chargedTrack=5.0, neutralTrack=5.0, jet=15.0),
        minDR=MinDR(jet=0.4, chargedTrack=0.4, neutralTrack=0.4),
        nStations=2, nIntersections=2, nTracks=1,
    )
    run_cfg = RunConfig(reweightLifetime=False, plotTrajectory=False)

    pipeline = (
        SelectionPipelineBuilder()
        .set_options(add_jets=True, compute_isolation=True, selection_mode=sel_mode)
        .build()
    )

    return pipeline, sel_cfg, run_cfg


def plot_bar(labels, values, title, out_pdf):
    plt.figure(figsize=(10, 4))
    plt.bar(labels, values, color="#4c78a8", width=1.0)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Events passing")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Run selection and plot cutflow from a bundle")
    ap.add_argument("--bundle", default="/usera/fs568/set-anubis/ALP_Z_Runs/ALP_Z_sampledfs_Scan_22_Run_3.pkl.gz", help="Path to gzipped bundle pickle")
    ap.add_argument("--sel-mode", default="standard", help="Selection mode (standard/2dv if available)")
    ap.add_argument("--name", default="sample", help="Sample name label for plots")
    ap.add_argument("--outdir", default="/usera/fs568/set-anubis/setanubis", help="Directory to save plots (default: outputs/)" )
    args = ap.parse_args()

    bundle_path = os.path.abspath(args.bundle)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    pipeline, sel_cfg, run_cfg = build_selection(sel_mode=args.sel_mode)
    source = EventsBundleSource.from_bundle_file(bundle_path)

    mgr = SelectionManager(pipeline)
    combined = mgr.run_many(named_sources=[(args.name, source)], sel_cfg=sel_cfg, run_cfg=run_cfg)

    out_dir = args.outdir or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..", "outputs"))
    os.makedirs(out_dir, exist_ok=True)

    ## Per-sample plot (first sample)
    #sample = combined.per_sample[0]
    ## Filter out weighted cuts (keys ending with "_weighted")
    #labels = [k for k in sample.cutFlow.keys() if not k.endswith("_weighted")]
    #values = [sample.cutFlow[k] for k in labels]
    #plot_bar(labels, values, f"Cutflow: {sample.name}", os.path.join(out_dir, f"cutflow_{sample.name}.png"))

    # SUM plot
    # Filter out weighted cuts
    labels_sum = [k for k in combined.cutflow_sum.keys() if not k.endswith("_weighted")]
    values_sum = [combined.cutflow_sum[k] for k in labels_sum]
    plot_bar(labels_sum, values_sum, "Cutflow: Ma = 10 GeV, CPhi = 0.00000316, fa = 1 TeV", os.path.join(out_dir, "cutflow_Scan_22_Run_3.pdf"))

    print(f"Saved plots in {out_dir}")


if __name__ == "__main__":
    main()
