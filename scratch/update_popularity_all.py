import json
from pathlib import Path

notebook_path = Path("/Users/ele/ao3-project2/popularity_analysis.ipynb")
with open(notebook_path, "r") as f:
    nb = json.load(f)

# Cell 2 source code with upstream cleaning
cell_2_source = [
    "# Clean and manipulate the dataset upstream using the AO3DataManipulator\n",
    "manipulator = AO3DataManipulator(df)\n",
    "manipulated_df = manipulator.run_all_manipulations()\n",
    "\n",
    "print(f\"Clean manipulated dataset size: {len(manipulated_df):,} rows.\")"
]

# Cell 7 source code with true bridge author logic
cell_7_source = [
    "import re\n",
    "import pandas as pd\n",
    "\n",
    "# Ensure author_str exists in manipulated_df\n",
    "if 'author_str' not in manipulated_df.columns:\n",
    "    manipulated_df['author_str'] = manipulated_df['parsed_authors'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')\n",
    "\n",
    "# 1. Define case-insensitive regex patterns for the two target fandoms\n",
    "marvel_pat = re.compile(r'marvel|avengers|mcu', re.IGNORECASE)\n",
    "teen_wolf_pat = re.compile(r'teen wolf', re.IGNORECASE)\n",
    "\n",
    "# 2. Filter the manipulated DataFrame using string masks on raw 'Fandom Tags'\n",
    "is_marvel = manipulated_df['Fandom Tags'].apply(lambda x: bool(marvel_pat.search(str(x))))\n",
    "is_teen_wolf = manipulated_df['Fandom Tags'].apply(lambda x: bool(teen_wolf_pat.search(str(x))))\n",
    "\n",
    "# 3. Exclude crossover works (works tagged with both Marvel AND Teen Wolf)\n",
    "is_crossover = is_marvel & is_teen_wolf\n",
    "marvel_only_df = manipulated_df[is_marvel & ~is_crossover].copy()\n",
    "teen_wolf_only_df = manipulated_df[is_teen_wolf & ~is_crossover].copy()\n",
    "\n",
    "# 4. Identify unique bridge authors writing in both fandoms (excluding crossovers)\n",
    "marvel_only_authors = set(marvel_only_df['author_str'].unique())\n",
    "teen_wolf_only_authors = set(teen_wolf_only_df['author_str'].unique())\n",
    "bridge_authors = marvel_only_authors.intersection(teen_wolf_only_authors)\n",
    "if 'Unknown' in bridge_authors:\n",
    "    bridge_authors.remove('Unknown')\n",
    "\n",
    "# 5. Filter works by bridge authors in both fandoms (excluding crossovers)\n",
    "marvel_bridge = marvel_only_df[marvel_only_df['author_str'].isin(bridge_authors)].copy()\n",
    "teen_wolf_bridge = teen_wolf_only_df[teen_wolf_only_df['author_str'].isin(bridge_authors)].copy()\n",
    "\n",
    "# 6. Create a combined filtered DataFrame containing only the works of these bridge authors in Marvel and Teen Wolf\n",
    "marvel_bridge['fandom_group'] = 'Marvel'\n",
    "teen_wolf_bridge['fandom_group'] = 'Teen Wolf'\n",
    "marvel_tw_bridge_df = pd.concat([marvel_bridge, teen_wolf_bridge], ignore_index=True)\n",
    "\n",
    "# Save to CSV for easy export and further deep-dive analysis\n",
    "marvel_tw_bridge_df.to_csv('marvel_tw_bridge_works.csv', index=False)\n",
    "print(\"Saved filtered bridge works to 'marvel_tw_bridge_works.csv'.\")\n",
    "\n",
    "# 7. Calculate overall baselines for TARGET_METRIC (using all Marvel/Teen Wolf works)\n",
    "marvel_overall_mean = manipulated_df[is_marvel][TARGET_METRIC].mean()\n",
    "marvel_overall_median = manipulated_df[is_marvel][TARGET_METRIC].median()\n",
    "\n",
    "tw_overall_mean = manipulated_df[is_teen_wolf][TARGET_METRIC].mean()\n",
    "tw_overall_median = manipulated_df[is_teen_wolf][TARGET_METRIC].median()\n",
    "\n",
    "# 8. Calculate bridge author work statistics for TARGET_METRIC (using non-crossover bridge works)\n",
    "marvel_bridge_mean = marvel_bridge[TARGET_METRIC].mean()\n",
    "marvel_bridge_median = marvel_bridge[TARGET_METRIC].median()\n",
    "marvel_bridge_ratio = marvel_bridge_median / marvel_overall_median if marvel_overall_median > 0 else 0\n",
    "\n",
    "tw_bridge_mean = teen_wolf_bridge[TARGET_METRIC].mean()\n",
    "tw_bridge_median = teen_wolf_bridge[TARGET_METRIC].median()\n",
    "tw_bridge_ratio = tw_bridge_median / tw_overall_median if tw_overall_median > 0 else 0\n",
    "\n",
    "# 9. Print summary statistics\n",
    "print(\"=\" * 60)\n",
    "print(f\"DEEP DIVE: MARVEL VS. TEEN WOLF (Metric: {TARGET_METRIC})\")\n",
    "print(\"=\" * 60)\n",
    "print(f\"Number of true bridge authors identified: {len(bridge_authors)}\")\n",
    "print(f\"Marvel bridge works: {len(marvel_bridge)} works\")\n",
    "print(f\"Teen Wolf bridge works: {len(teen_wolf_bridge)} works\")\n",
    "print(f\"Total bridge works in combined DataFrame: {len(marvel_tw_bridge_df)}\")\n",
    "print(\"-\" * 60)\n",
    "print(f\"Marvel Fandom Overall Median {TARGET_METRIC}: {marvel_overall_median:.1f} (Mean: {marvel_overall_mean:.1f})\")\n",
    "print(f\"Bridge Authors' Marvel Works Median {TARGET_METRIC}: {marvel_bridge_median:.1f} (Mean: {marvel_bridge_mean:.1f})\")\n",
    "print(f\"Relative Popularity Ratio in Marvel: {marvel_bridge_ratio:.4f}\")\n",
    "print(\"-\" * 60)\n",
    "print(f\"Teen Wolf Fandom Overall Median {TARGET_METRIC}: {tw_overall_median:.1f} (Mean: {tw_overall_mean:.1f})\")\n",
    "print(f\"Bridge Authors' Teen Wolf Works Median {TARGET_METRIC}: {tw_bridge_median:.1f} (Mean: {tw_bridge_mean:.1f})\")\n",
    "print(f\"Relative Popularity Ratio in Teen Wolf: {tw_bridge_ratio:.4f}\")\n",
    "print(\"=\" * 60)\n",
    "\n",
    "# 10. Show a preview of the bridge works\n",
    "print(\"\\nPreview of Marvel vs. Teen Wolf Bridge Works:\")\n",
    "print(marvel_tw_bridge_df[['Title', 'author_str', 'fandom_group', TARGET_METRIC, 'Words']].head(15))\n",
    "\n",
    "# 11. Render a premium comparison visualization\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))\n",
    "\n",
    "# Subplot 1: Absolute Medians comparison\n",
    "categories = ['Marvel Overall', \"Marvel Bridge\", 'Teen Wolf Overall', \"Teen Wolf Bridge\"]\n",
    "values = [marvel_overall_median, marvel_bridge_median, tw_overall_median, tw_bridge_median]\n",
    "colors = ['#b2bec3', '#ff7675', '#dfe6e9', '#74b9ff']\n",
    "\n",
    "bars1 = ax1.bar(categories, values, color=colors, edgecolor='none', width=0.6)\n",
    "ax1.set_ylabel(f'Median {TARGET_METRIC}', fontsize=12, labelpad=8)\n",
    "ax1.set_title(f'Absolute Median {TARGET_METRIC} Comparison', fontsize=13, weight='bold', pad=12)\n",
    "ax1.grid(True, axis='y', linestyle='--', alpha=0.5)\n",
    "\n",
    "# Add value labels on top of the bars\n",
    "for bar in bars1:\n",
    "    height = bar.get_height()\n",
    "    ax1.annotate(f'{height:.0f}',\n",
    "                 xy=(bar.get_x() + bar.get_width() / 2, height),\n",
    "                 xytext=(0, 3),  # 3 points vertical offset\n",
    "                 textcoords=\"offset points\",\n",
    "                 ha='center', va='bottom', fontsize=10, weight='bold')\n",
    "\n",
    "# Subplot 2: Relative Popularity Ratio comparison\n",
    "ratios = [marvel_bridge_ratio, tw_bridge_ratio]\n",
    "ratio_labels = ['Marvel', 'Teen Wolf']\n",
    "ratio_colors = ['#ff7675', '#74b9ff']\n",
    "\n",
    "bars2 = ax2.bar(ratio_labels, ratios, color=ratio_colors, edgecolor='none', width=0.4)\n",
    "ax2.axhline(1.0, color='#e53e3e', linestyle='--', linewidth=1.5, alpha=0.8, label='Fandom Baseline (1.0)')\n",
    "ax2.set_ylabel(f'Popularity Ratio vs Fandom Median', fontsize=12, labelpad=8)\n",
    "ax2.set_title(f'Relative Popularity Ratio (Bridge / Fandom Median)', fontsize=13, weight='bold', pad=12)\n",
    "ax2.grid(True, axis='y', linestyle='--', alpha=0.5)\n",
    "ax2.legend(loc='lower left')\n",
    "\n",
    "# Add ratio labels on top of the bars\n",
    "for bar in bars2:\n",
    "    height = bar.get_height()\n",
    "    ax2.annotate(f'{height:.4f}',\n",
    "                 xy=(bar.get_x() + bar.get_width() / 2, height),\n",
    "                 xytext=(0, 3),  # 3 points vertical offset\n",
    "                 textcoords=\"offset points\",\n",
    "                 ha='center', va='bottom', fontsize=10, weight='bold')\n",
    "\n",
    "plt.suptitle(f'Deep Dive: Bridge Authors in Marvel vs. Teen Wolf ({TARGET_METRIC})', fontsize=15, weight='bold', y=0.98)\n",
    "plt.tight_layout()\n",
    "plt.savefig('docs/marvel_teen_wolf_deep_dive.png', dpi=300, bbox_inches='tight')\n",
    "plt.show()"
]

# Update cell 2 and cell 7
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        if "Clean manipulated dataset size" in source_str or "Clean dataset (matching the main analysis notebook filtering pipeline)" in source_str:
            cell["source"] = cell_2_source
            print("Cell 2 updated.")
            
        if "Deep Dive: Marvel VS. Teen Wolf" in source_str or "DEEP DIVE: MARVEL VS. TEEN WOLF" in source_str:
            cell["source"] = cell_7_source
            print("Cell 7 updated.")

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)
print("Finished updating popularity_analysis.ipynb successfully!")
