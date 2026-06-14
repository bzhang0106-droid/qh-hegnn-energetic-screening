from __future__ import annotations
import gzip, json
from pathlib import Path
import pandas as pd

ROOT = Path('/home/gma/bzhang/bzhang/Workflow2.0')
INP = ROOT / 'results/final_global_top20/Final_Global_Top20_Retrosynthesis_Input.smi'
RAW = ROOT / 'results/final_global_top20/aizynthfinder/final_global_top20_aizynthfinder_results_v2.json.gz'
TOP20 = ROOT / 'results/final_global_top20/Table_Final_Global_Top20_Structure_Property_Synthesizability_10D.csv'
OUT = ROOT / 'results/final_global_top20/aizynthfinder/Table_Final_Global_Top20_AiZynthFinder_Route_Summary_20260606.csv'
MERGED = ROOT / 'results/final_global_top20/Table_Final_Global_Top20_Structure_Property_Synthesizability_AiZynthFinder_10D.csv'
PKG_MAIN = ROOT / 'manuscript_npJ/final_submission_package_AL08_20260605/main_tables/Table_Final_Global_Top20_Structure_Property_Synthesizability_AiZynthFinder_10D.csv'
PKG_SI = ROOT / 'manuscript_npJ/final_submission_package_AL08_20260605/si_tables/Table_S_Final_Global_Top20_AiZynthFinder_Route_Summary_20260606.csv'

rows = []
for i, line in enumerate(INP.read_text().splitlines()):
    if not line.strip():
        continue
    parts = line.strip().split(maxsplit=1)
    rows.append({'index': i, 'Input_SMILES': parts[0], 'Molecule': parts[1] if len(parts) > 1 else ''})
input_df = pd.DataFrame(rows)

with gzip.open(RAW, 'rt') as fh:
    raw = json.load(fh)
res = pd.DataFrame(raw['data'])
res['index'] = res['index'].astype(int)
summary = input_df.merge(res, on='index', how='left')
keep = [
    'index','Molecule','Input_SMILES','target','is_solved','search_time','first_solution_time',
    'first_solution_iteration','number_of_nodes','number_of_routes','number_of_solved_routes',
    'top_score','number_of_steps','number_of_precursors','number_of_precursors_in_stock',
    'precursors_in_stock','precursors_not_in_stock','precursors_availability','policy_used_counts'
]
summary = summary[keep]
summary.insert(0, 'Final_Global_Rank', summary['index'] + 1)
summary['AiZynthFinder_Status'] = summary['is_solved'].map({True:'solved_to_zinc_stock', False:'not_solved_to_zinc_stock'})
summary['AiZynthFinder_Config'] = 'AiZynthFinder 4.4.1; public USPTO policy/filter; ZINC stock; max_transforms default; max_iterations default 100; nproc=12'
OUT.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT, index=False)

top = pd.read_csv(TOP20)
merged = top.merge(summary.drop(columns=['index']), on=['Final_Global_Rank','Molecule'], how='left')
# Update external status without erasing original screening evidence.
if 'External_Retrosynthesis_Status' in merged.columns:
    merged['External_Retrosynthesis_Status'] = merged['AiZynthFinder_Status'].fillna(merged['External_Retrosynthesis_Status'])
merged.to_csv(MERGED, index=False)
PKG_MAIN.parent.mkdir(parents=True, exist_ok=True)
PKG_SI.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(PKG_MAIN, index=False)
summary.to_csv(PKG_SI, index=False)
print('summary_rows', len(summary), 'solved', int(summary['is_solved'].fillna(False).sum()))
print('summary', OUT)
print('merged', MERGED)
print(summary[['Final_Global_Rank','Molecule','is_solved','number_of_routes','number_of_solved_routes','top_score','number_of_steps','number_of_precursors_in_stock']].to_string(index=False))
