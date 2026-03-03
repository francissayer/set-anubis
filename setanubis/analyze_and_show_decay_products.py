#!/usr/bin/env python3
"""Analyze immediate-decay and final-state PDG summaries from set-anubis pickles.

Configure `INPUT_PATH` and `TARGET_PDG` below and run the script. The script prints:
- top final-state PDGs (post-hadronization)
- initial-decay summary for the target parent (using `LLPchildren` when available)
- final-state summary restricted to events containing the parent
"""
from collections import Counter
import gzip
import pickle
import sys
import ast

# ===== Configuration =====
INPUT_PATH = '/usera/fs568/set-anubis/ALP_Z_Runs/ALP_Z_sampledfs_Scan_31_Run_1.pkl.gz'
TARGET_PDG = 9000005
# If None, the script will auto-detect DataFrames inside the pickle
PARENTS_KEY = 'LLPs'          # typically 'LLPs'
FINALSTATES_KEY = 'finalStates'  # typically 'finalStates'
# ==========================
# ==========================
# ==========================


def load_pickle(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        return pickle.load(f)


def pdg_name(pdg):
    try:
        from particle import Particle
        p = Particle.from_pdgid(int(pdg))
        return p.name if getattr(p, 'name', None) else str(pdg)
    except Exception:
        fallback = {
            22: 'gamma', 11: 'e-', -11: 'e+', 13: 'mu-', -13: 'mu+',
            111: 'pi0', 211: 'pi+', -211: 'pi-', 321: 'K+', -321: 'K-',
            310: 'K0S', 130: 'K0L', 221: 'eta', 331: "eta'", 333: 'phi'
        }
        return fallback.get(int(pdg), str(pdg))


def choose_dataframes(obj, parents_key=None, final_key=None):
    import pandas as pd

    parents = None
    final = None
    if isinstance(obj, dict):
        if parents_key and parents_key in obj and isinstance(obj[parents_key], pd.DataFrame):
            parents = obj[parents_key]
        if final_key and final_key in obj and isinstance(obj[final_key], pd.DataFrame):
            final = obj[final_key]

        # fallback searches
        if parents is None:
            for k, v in obj.items():
                if isinstance(v, pd.DataFrame) and 'PID' in v.columns and 'childrenIndices' in v.columns:
                    parents = v; break
        if final is None:
            for k, v in obj.items():
                if isinstance(v, pd.DataFrame) and 'particleIndex' in v.columns:
                    final = v; break

    elif isinstance(obj, pd.DataFrame):
        # single dataframe; treat as finalStates
        final = obj

    return parents, final


def detect_pid_column(df):
    candidates = ['PID', 'pid', 'PDG', 'pdg', 'Id', 'ID']
    for c in candidates:
        if c in df.columns:
            return c
    lc = {col.lower(): col for col in df.columns}
    for c in ['pid', 'pdg', 'id']:
        if c in lc:
            return lc[c]
    raise KeyError('No PID/PDG column found in DataFrame columns: ' + ','.join(map(str, df.columns)))


def parse_indices(val):
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    s = str(val).strip()
    if s == '' or s.lower() == 'nan' or s == '[]':
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [int(x) for x in parsed]
    except Exception:
        pass
    if ',' in s:
        parts = [p.strip().strip('[]') for p in s.split(',')]
        out = []
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                pass
        return out
    try:
        return [int(s)]
    except Exception:
        return []


# removed unused helpers to keep the script compact and fast


def analyze_children_table(final_df):
    pid_col = detect_pid_column(final_df)
    series = final_df[pid_col].astype(int)
    counts = series.value_counts()
    total = len(final_df)
    return counts, total


def main():
    try:
        obj = load_pickle(INPUT_PATH)
    except Exception as e:
        print('Failed to load pickle:', e, file=sys.stderr)
        sys.exit(2)

    parents_df, final_df = choose_dataframes(obj, PARENTS_KEY, FINALSTATES_KEY)
    # try to find LLPchildren table in the pickle (many pickles store immediate children here)
    import pandas as pd
    llpchildren_df = None
    if isinstance(obj, dict):
        if 'LLPchildren' in obj and isinstance(obj['LLPchildren'], pd.DataFrame):
            llpchildren_df = obj['LLPchildren']
        else:
            for k, v in obj.items():
                if isinstance(v, pd.DataFrame) and 'LLPindex' in v.columns and 'particleIndex' in v.columns:
                    llpchildren_df = v
                    break
    if final_df is None:
        print('Could not detect finalStates DataFrame in pickle', file=sys.stderr); sys.exit(3)

    # --- Summary: final-states table overall (from selected finalStates DataFrame) ---
    counts, total = analyze_children_table(final_df)
    print(f'Final-states table rows: {total}')
    print('\nTop final-state PDGs (post-hadronization):')
    for pid, cnt in counts.head(30).items():
        name = pdg_name(pid)
        print(f' PDG {int(pid):6d} ({name:12s}): {int(cnt):6d}  fraction={cnt/total:.2%}')

    # --- Initial decays: what each parent ALP decays into (immediate daughters) ---
    if parents_df is None:
        print('\nNo parents DataFrame detected; cannot summarize initial decays.', file=sys.stderr)
        return

    # select parent rows matching TARGET_PDG
    try:
        p_pid_col = 'PID' if 'PID' in parents_df.columns else detect_pid_column(parents_df)
    except Exception:
        p_pid_col = None

    if p_pid_col is None:
        print('\nCould not detect PID column in parents DataFrame; skipping initial-decay summary.', file=sys.stderr)
        return

    parents_sel = parents_df[parents_df[p_pid_col].astype(int) == int(TARGET_PDG)]
    n_parents = len(parents_sel)
    if n_parents == 0:
        print(f'\nNo parents with PDG {TARGET_PDG} found in parents DataFrame.')
        return

    # coerce numeric columns in final_df for fast merges
    try:
        fid = detect_pid_column(final_df)
        final_df[fid] = pd.to_numeric(final_df[fid], errors='coerce').fillna(0).astype(int)
    except Exception:
        fid = None
    for c in ('eventNumber', 'particleIndex'):
        if c in final_df.columns:
            final_df[c] = pd.to_numeric(final_df[c], errors='coerce').fillna(-1).astype(int)

    # final_map removed (not used)

    # prepare LLPchildren mapping (vectorized)
    llp_map = None
    if llpchildren_df is not None:
        # coerce common numeric columns
        try:
            llp_pid_col = detect_pid_column(llpchildren_df)
        except Exception:
            llp_pid_col = None
        for c in ('eventNumber', 'particleIndex'):
            if c in llpchildren_df.columns:
                llpchildren_df[c] = pd.to_numeric(llpchildren_df[c], errors='coerce').fillna(-1).astype(int)
        # detect a column linking child -> LLP parent (LLPindex or similar)
        llp_index_col = None
        for c in llpchildren_df.columns:
            if 'llp' in c.lower() and 'index' in c.lower():
                llp_index_col = c
                break
        if llp_index_col is None and 'LLPindex' in llpchildren_df.columns:
            llp_index_col = 'LLPindex'
        if llp_index_col is not None:
            # coerce link column
            llpchildren_df[llp_index_col] = pd.to_numeric(llpchildren_df[llp_index_col], errors='coerce').fillna(-1).astype(int)
            llp_map = {
                'df': llpchildren_df,
                'pid_col': llp_pid_col,
                'link_col': llp_index_col
            }

    # Vectorized initial-decay computation
    initial_counter = Counter()
    missing = 0
    events_set = set(parents_sel['eventNumber'].astype(int).unique())
    # Fast path: use llp_map if available
    if llp_map is not None:
        df = llp_map['df']
        pid_col = llp_map['pid_col']
        link_col = llp_map['link_col']
        # try to use parents_sel.index as LLPindex; fall back to a parent id column
        try:
            parent_idx_series = parents_sel.index.astype(int)
        except Exception:
            # try common parent index columns
            if 'particleIndex' in parents_sel.columns:
                parent_idx_series = parents_sel['particleIndex'].astype(int)
            else:
                parent_idx_series = pd.Series([], dtype=int)

        if len(parent_idx_series) > 0:
            sel_children = df[df[link_col].isin(parent_idx_series)]
            if pid_col is not None and pid_col in sel_children.columns:
                counts = sel_children[pid_col].astype(int).value_counts()
                for pid, cnt in counts.items():
                    initial_counter[int(pid)] += int(cnt)
    else:
        # explode childrenIndices for selected parents and merge with final_df
        children_series = parents_sel['childrenIndices'].dropna().apply(parse_indices)
        if children_series.empty:
            missing = len(parents_sel)
        else:
            children_exp = children_series.explode().rename('childIndex')
            children_exp = children_exp.to_frame().join(parents_sel[['eventNumber']], how='left')
            children_exp = children_exp.dropna(subset=['childIndex'])
            children_exp['childIndex'] = pd.to_numeric(children_exp['childIndex'], errors='coerce').astype('Int64')
            # merge with final_df on eventNumber & particleIndex
            merge_left = children_exp.rename(columns={'childIndex': 'particleIndex'})
            merged = merge_left.merge(final_df[[c for c in ('eventNumber', 'particleIndex') if c in final_df.columns] + ([fid] if fid else [])], on=['eventNumber', 'particleIndex'], how='left')
            if fid and fid in merged.columns:
                counts = merged[fid].dropna().astype(int).value_counts()
                for pid, cnt in counts.items():
                    initial_counter[int(pid)] += int(cnt)
            missing = merged['particleIndex'].isna().sum() if 'particleIndex' in merged.columns else 0

    total_initial_children = sum(initial_counter.values())
    print(f'\nInitial-decay summary for PDG {TARGET_PDG} (parents found: {n_parents}):')
    if total_initial_children == 0:
        print(' No child entries found (childrenIndices missing or children not present in finalStates).')
    else:
        for pid, cnt in initial_counter.most_common(30):
            name = pdg_name(pid)
            print(f' PDG {int(pid):6d} ({name:12s}): {int(cnt):6d}  fraction_of_initial_children={cnt/total_initial_children:.2%}  avg_per_parent={cnt/n_parents:.3f}')
    if missing:
        print(f' {missing} child indices referenced by parents were not found in the finalStates table.')

    # --- Final-states restricted to events with the selected parents ---
    sel_final = final_df[final_df['eventNumber'].isin(events_set)]
    sel_total = len(sel_final)
    sel_counts = sel_final[detect_pid_column(sel_final)].astype(int).value_counts()
    print(f'\nFinal-states summary restricted to events with PDG {TARGET_PDG} parents (events: {len(events_set)}):')
    for pid, cnt in sel_counts.head(30).items():
        name = pdg_name(pid)
        print(f' PDG {int(pid):6d} ({name:12s}): {int(cnt):6d}  fraction_of_sel_final={cnt/sel_total:.2%}  avg_per_event={cnt/len(events_set):.3f}')


if __name__ == '__main__':
    main()
