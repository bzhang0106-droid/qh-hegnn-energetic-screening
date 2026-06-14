from __future__ import annotations
from pathlib import Path
import pandas as pd
ROOT=Path('/home/gma/bzhang/bzhang/Workflow2.0')
top=ROOT/'results/final_global_top20/Table_Final_Global_Top20_Structure_Property_Synthesizability_AiZynthFinder_10D.csv'
ai=ROOT/'results/final_global_top20/aizynthfinder/Table_Final_Global_Top20_AiZynthFinder_Route_Summary_20260606.csv'
ask=ROOT/'results/final_global_top20/askcos/Table_Final_Global_Top20_ASKCOS_MCTS_Summary_20260606.csv'
out=ROOT/'results/final_global_top20/Table_Final_Global_Top20_Structure_Property_Synthesizability_ExternalRetrosynthesis_10D.csv'
summary_out=ROOT/'results/final_global_top20/Table_Final_Global_Top20_External_Retrosynthesis_Summary_20260606.csv'
pkg_main=ROOT/'manuscript_npJ/final_submission_package_AL08_20260605/main_tables/Table_Final_Global_Top20_Structure_Property_Synthesizability_ExternalRetrosynthesis_10D.csv'
pkg_si=ROOT/'manuscript_npJ/final_submission_package_AL08_20260605/si_tables/Table_S_Final_Global_Top20_External_Retrosynthesis_Summary_20260606.csv'
for p in [top, ai, ask]:
    if not p.exists(): raise FileNotFoundError(p)
df=pd.read_csv(top)
ai_df=pd.read_csv(ai)
ask_df=pd.read_csv(ask)
# Avoid duplicate detailed columns already merged from AiZynthFinder; retain compact route fields.
ai_keep=['Final_Global_Rank','Molecule','AiZynthFinder_Status','search_time','number_of_nodes','number_of_routes','number_of_solved_routes','top_score','number_of_steps','number_of_precursors','number_of_precursors_in_stock','precursors_in_stock','precursors_not_in_stock','AiZynthFinder_Config']
ask_keep=['Final_Global_Rank','Molecule','ASKCOS_Status','ASKCOS_Total_Iterations','ASKCOS_Total_Chemicals','ASKCOS_Total_Reactions','ASKCOS_Total_Templates','ASKCOS_Total_Paths','ASKCOS_Pathways_Returned','ASKCOS_First_Path_Time_s','ASKCOS_Build_Time_s','ASKCOS_Settings','ASKCOS_Query_File','ASKCOS_Response_File']
ai_comp=ai_df[ai_keep].rename(columns={
    'search_time':'AiZynthFinder_Search_Time_s',
    'number_of_nodes':'AiZynthFinder_Number_Of_Nodes',
    'number_of_routes':'AiZynthFinder_Number_Of_Routes',
    'number_of_solved_routes':'AiZynthFinder_Number_Of_Solved_Routes',
    'top_score':'AiZynthFinder_Top_Score',
    'number_of_steps':'AiZynthFinder_Top_Route_Steps',
    'number_of_precursors':'AiZynthFinder_Top_Route_Precursors',
    'number_of_precursors_in_stock':'AiZynthFinder_Top_Route_Precursors_In_ZINC_Stock',
    'precursors_in_stock':'AiZynthFinder_Precursors_In_Stock',
    'precursors_not_in_stock':'AiZynthFinder_Precursors_Not_In_Stock',
})
ask_comp=ask_df[ask_keep]
# Drop older duplicate AiZynth columns from df before re-merging compact names.
drop_cols=[c for c in df.columns if c.startswith('AiZynthFinder_') or c in ['search_time','number_of_nodes','number_of_routes','number_of_solved_routes','top_score','number_of_steps','number_of_precursors','number_of_precursors_in_stock','precursors_in_stock','precursors_not_in_stock']]
df=df.drop(columns=drop_cols, errors='ignore')
merged=df.merge(ai_comp,on=['Final_Global_Rank','Molecule'],how='left').merge(ask_comp,on=['Final_Global_Rank','Molecule'],how='left')
merged['External_Retrosynthesis_Overall_Status']='queried_no_complete_buyable_route_found'
merged.loc[(merged['AiZynthFinder_Number_Of_Solved_Routes'].fillna(0)>0) | (merged['ASKCOS_Total_Paths'].fillna(0)>0),'External_Retrosynthesis_Overall_Status']='complete_or_candidate_route_found'
merged['Submission_Synthesizability_Claim_Status']=merged['External_Retrosynthesis_Overall_Status'].map({
    'complete_or_candidate_route_found':'route_tool_candidate_available_requires_chemist_review',
    'queried_no_complete_buyable_route_found':'screening_only_route_tools_did_not_solve'
})
merged.to_csv(out,index=False)
pkg_main.parent.mkdir(parents=True, exist_ok=True); pkg_si.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(pkg_main,index=False)
summary_cols=['Final_Global_Rank','Molecule','Input_SMILES','AiZynthFinder_Status','AiZynthFinder_Number_Of_Routes','AiZynthFinder_Number_Of_Solved_Routes','AiZynthFinder_Top_Score','AiZynthFinder_Top_Route_Steps','AiZynthFinder_Top_Route_Precursors_In_ZINC_Stock','ASKCOS_Status','ASKCOS_Total_Iterations','ASKCOS_Total_Reactions','ASKCOS_Total_Templates','ASKCOS_Total_Paths','ASKCOS_Pathways_Returned','External_Retrosynthesis_Overall_Status','Submission_Synthesizability_Claim_Status']
# Input_SMILES may be named by previous merge.
if 'Input_SMILES' not in merged.columns and 'SMILES' in merged.columns:
    merged['Input_SMILES']=merged['SMILES']
summary=merged[[c for c in summary_cols if c in merged.columns]]
summary.to_csv(summary_out,index=False)
summary.to_csv(pkg_si,index=False)
print('merged_rows',len(merged))
print('aizynth_solved', int((merged['AiZynthFinder_Number_Of_Solved_Routes'].fillna(0)>0).sum()))
print('askcos_paths', int((merged['ASKCOS_Total_Paths'].fillna(0)>0).sum()))
print('out',out)
print(summary.to_string(index=False))
