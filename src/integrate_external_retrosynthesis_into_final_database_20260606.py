from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path('/home/gma/bzhang/bzhang/Workflow2.0')
DB = ROOT / 'data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv'
EXT = ROOT / 'results/final_global_top20/Table_Final_Global_Top20_External_Retrosynthesis_Summary_20260606.csv'
OUT_REPORT = ROOT / 'manuscript_npJ/Final_Database_External_Retrosynthesis_Integration_Report_20260606.md'
PKG = ROOT / 'manuscript_npJ/final_submission_package_AL08_20260605'

external_cols = [
    'External_Retrosynthesis_Query_Scope',
    'External_Retrosynthesis_Overall_Status',
    'Submission_Synthesizability_Claim_Status',
    'AiZynthFinder_Status',
    'AiZynthFinder_Number_Of_Routes',
    'AiZynthFinder_Number_Of_Solved_Routes',
    'AiZynthFinder_Top_Score',
    'AiZynthFinder_Top_Route_Steps',
    'AiZynthFinder_Top_Route_Precursors_In_ZINC_Stock',
    'ASKCOS_Status',
    'ASKCOS_Total_Iterations',
    'ASKCOS_Total_Reactions',
    'ASKCOS_Total_Templates',
    'ASKCOS_Total_Paths',
    'ASKCOS_Pathways_Returned',
]

db = pd.read_csv(DB)
ext = pd.read_csv(EXT)
rows_before, cols_before = db.shape
if rows_before != 5432:
    raise RuntimeError(f'Unexpected DB row count before integration: {rows_before}')
required_db = {'Final_Detonation_Rank','Molecule'}
required_ext = {'Final_Global_Rank','Molecule'}
missing_db = required_db - set(db.columns)
missing_ext = required_ext - set(ext.columns)
if missing_db or missing_ext:
    raise RuntimeError(f'Missing key columns db={missing_db} ext={missing_ext}')

# Check that current globally sorted DB top20 and external top20 are the same molecules in the same rank order.
db_top = db.loc[db['Final_Detonation_Rank'].between(1,20), ['Final_Detonation_Rank','Molecule']].copy()
ext_key = ext[['Final_Global_Rank','Molecule']].copy().rename(columns={'Final_Global_Rank':'Final_Detonation_Rank'})
check = db_top.merge(ext_key, on=['Final_Detonation_Rank','Molecule'], how='outer', indicator=True)
if not (check['_merge'] == 'both').all() or len(check) != 20:
    raise RuntimeError('Top20 rank/molecule mismatch between sorted final DB and external retrosynthesis table:\n' + check.to_string(index=False))

# Remove stale integration columns if rerun, then merge current external evidence.
db = db.drop(columns=[c for c in external_cols if c in db.columns], errors='ignore')
ext2 = ext.rename(columns={'Final_Global_Rank':'Final_Detonation_Rank'}).copy()
ext2['External_Retrosynthesis_Query_Scope'] = 'final_global_top20_only'
keep = ['Final_Detonation_Rank','Molecule'] + external_cols
for c in keep:
    if c not in ext2.columns:
        ext2[c] = pd.NA
ext2 = ext2[keep]
merged = db.merge(ext2, on=['Final_Detonation_Rank','Molecule'], how='left', validate='one_to_one')
if len(merged) != rows_before:
    raise RuntimeError(f'Row count changed after integration: {rows_before} -> {len(merged)}')

# Keep integer-like external columns as nullable where possible.
for c in ['AiZynthFinder_Number_Of_Routes','AiZynthFinder_Number_Of_Solved_Routes','AiZynthFinder_Top_Route_Steps','AiZynthFinder_Top_Route_Precursors_In_ZINC_Stock','ASKCOS_Total_Iterations','ASKCOS_Total_Reactions','ASKCOS_Total_Templates','ASKCOS_Total_Paths','ASKCOS_Pathways_Returned']:
    if c in merged.columns:
        merged[c] = pd.to_numeric(merged[c], errors='coerce').astype('Int64')

# In-place official DB update per project convention: no separate backup file.
merged.to_csv(DB, index=False)
# Sync final package supplementary data and compact top20-aware table.
(PKG / 'supplementary_data').mkdir(parents=True, exist_ok=True)
(PKG / 'main_tables').mkdir(parents=True, exist_ok=True)
merged.to_csv(PKG / 'supplementary_data/Supplementary_Data_1_Final_AL08_Database.csv', index=False)
merged.loc[merged['Final_Detonation_Rank'].between(1,20)].to_csv(PKG / 'main_tables/Table_Final_Global_Top20_From_Final_Database_With_ExternalRetrosynthesis_20260606.csv', index=False)

nonempty = {c: int(merged[c].notna().sum()) for c in external_cols}
summary = [
    '# Final Database External Retrosynthesis Integration Report 2026-06-06',
    '',
    f'Official DB: `{DB.relative_to(ROOT)}`',
    '',
    f'- Rows before/after: {rows_before} / {len(merged)}',
    f'- Columns before/after: {cols_before} / {merged.shape[1]}',
    '- Integration scope: final global Top20 only; all non-Top20 molecules intentionally have blank external-retrosynthesis fields.',
    '- Top20 key check: `Final_Detonation_Rank` + `Molecule` matched exactly against external retrosynthesis summary.',
    '- No molecules were deleted or re-ranked during this integration.',
    '',
    '## Added/Refreshed Columns',
    '',
]
for c in external_cols:
    summary.append(f'- `{c}`: non-null rows = {nonempty[c]}')
summary += [
    '',
    '## Interpretation For Manuscript',
    '',
    'ASKCOS and AiZynthFinder provide an external retrosynthesis-tool audit for the final global Top20. Both tools completed queries for all 20 molecules, but neither identified complete buyable routes under the tested public settings. Therefore these fields support a cautious claim: the top candidates are high-performing computational leads with explicit route-planning risk, not experimentally or route-proven synthesizable molecules.',
]
OUT_REPORT.write_text('\n'.join(summary) + '\n')
print('\n'.join(summary))
