import json
from pathlib import Path

notebook_path = Path("/Users/ele/ao3-project2/analysis.ipynb")
with open(notebook_path, "r") as f:
    nb = json.load(f)

# Cell 2 source code with URL duplicate removal and explanations
cell_2_source = [
    "# print(f\"Original record count: {len(df):,}\\n\")\n",
    "\n",
    "# Filter 0: Drop duplicate works by URL\n",
    "df_uniq = df.drop_duplicates(subset=['URL'])\n",
    "count_uniq = len(df_uniq)\n",
    "removed_uniq = len(df) - count_uniq\n",
    "# print(f\"0. After removing duplicate works by URL:\")\n",
    "# print(f\"   Remaining: {count_uniq:,} (Removed: {removed_uniq:,})\")\n",
    "\n",
    "# Filter 1: Author not empty and not ['orphan_account']\n",
    "df_filtered_1 = df_uniq[df_uniq['parsed_authors'].apply(lambda x: len(x) > 0 and x != ['orphan_account'])]\n",
    "count_1 = len(df_filtered_1)\n",
    "removed_1 = count_uniq - count_1\n",
    "# print(f\"1. After removing empty authors and 'orphan_account' works:\")\n",
    "# print(f\"   Remaining: {count_1:,} (Removed: {removed_1:,})\")\n",
    "\n",
    "# Filter 2: Filter out works with more than one author\n",
    "df_filtered_2 = df_filtered_1[df_filtered_1['parsed_authors'].apply(lambda x: len(x) == 1)]\n",
    "count_2 = len(df_filtered_2)\n",
    "removed_2 = count_1 - count_2\n",
    "# print(f\"2. After filtering out works with multiple authors:\")\n",
    "# print(f\"   Remaining: {count_2:,} (Removed: {removed_2:,})\")\n",
    "\n",
    "# Filter 3: Filter out works with more than 5 fandom tags\n",
    "df_filtered_3 = df_filtered_2[df_filtered_2['parsed_fandoms'].apply(lambda x: len(x) <= 5)]\n",
    "count_3 = len(df_filtered_3)\n",
    "removed_3 = count_2 - count_3\n",
    "# print(f\"3. After filtering out works with > 5 fandom tags:\")\n",
    "# print(f\"   Remaining: {count_3:,} (Removed: {removed_3:,})\")\n"
]

# Cell 5 source code calling manipulator directly on the raw dataset
cell_5_source = [
    "from ao3_manipulator import AO3DataManipulator\n",
    "\n",
    "# Initialize the manipulator directly on the raw dataset so all filters run upstream!\n",
    "manipulator = AO3DataManipulator(df)\n",
    "\n",
    "# Run all manipulations (this cleans the dataset upstream and engineers features)\n",
    "manipulated_df = manipulator.run_all_manipulations()\n",
    "\n",
    "# Assign to final_analysis_df for cascading compatibility\n",
    "final_analysis_df = manipulated_df\n"
]

# Update cell 2 and cell 5
updated_c2 = False
updated_c5 = False

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        if "# Filter 1: Author not empty and not ['orphan_account']" in source_str and "df_filtered_3 = df_filtered_2" in source_str:
            cell["source"] = cell_2_source
            updated_c2 = True
            print("Cell 2 in analysis.ipynb matched and updated.")
            
        if "manipulator = AO3DataManipulator(final_analysis_df)" in source_str:
            cell["source"] = cell_5_source
            updated_c5 = True
            print("Cell 5 in analysis.ipynb matched and updated.")

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Finished updating analysis.ipynb. Success: Cell 2={updated_c2}, Cell 5={updated_c5}")
