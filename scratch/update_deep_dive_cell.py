import json
from pathlib import Path

notebook_path = Path("/Users/ele/ao3-project2/popularity_analysis.ipynb")
with open(notebook_path, "r") as f:
    nb = json.load(f)

# Find the deep dive markdown and code cells and replace them
found_markdown = False
found_code = False

# New markdown and code sources
markdown_source = [
    "## 7. Deep Dive: Marvel (Mega-Fandom) vs. Teen Wolf (Small Fandom) Crossover\n",
    "\n",
    "In this section, we perform a deep-dive analysis on the bridge authors who write for both the **Marvel Fandom** (a mega-fandom) and the **Teen Wolf Fandom** (a relatively small fandom).\n",
    "\n",
    "As noted during fandom clustering, connected-component clustering at a 0.5 threshold groups `'Teen Wolf (TV)'` under the `'Marvel Cinematic Universe'` cluster due to structural crossover chaining. To ensure absolute precision and keep the two fandoms separate, we filter works using case-insensitive string matching directly on the raw `'Fandom Tags'` column.\n",
    "\n",
    "Specifically, we:\n",
    "1. Filter works matching `marvel|avengers|mcu` for Marvel, and `teen wolf` for Teen Wolf.\n",
    "2. Identify the unique bridge authors writing in both groups.\n",
    "3. Extract all works by these bridge authors in both fandoms into a combined, clean DataFrame (`marvel_tw_bridge_df`) and export it to a CSV file.\n",
    "4. Calculate overall fandom baselines and compare them to the bridge authors' works using the active `TARGET_METRIC` (e.g., Hits, Kudos, or Comments)."
]

code_source = [
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
    "marvel_df = manipulated_df[is_marvel].copy()\n",
    "teen_wolf_df = manipulated_df[is_teen_wolf].copy()\n",
    "\n",
    "# 3. Identify unique bridge authors writing in both fandoms\n",
    "marvel_authors = set(marvel_df['author_str'].unique())\n",
    "teen_wolf_authors = set(teen_wolf_df['author_str'].unique())\n",
    "bridge_authors = marvel_authors.intersection(teen_wolf_authors)\n",
    "if 'Unknown' in bridge_authors:\n",
    "    bridge_authors.remove('Unknown')\n",
    "\n",
    "# 4. Filter works by bridge authors in both fandoms\n",
    "marvel_bridge = marvel_df[marvel_df['author_str'].isin(bridge_authors)].copy()\n",
    "teen_wolf_bridge = teen_wolf_df[teen_wolf_df['author_str'].isin(bridge_authors)].copy()\n",
    "\n",
    "# 5. Create a combined filtered DataFrame containing only the works of these bridge authors in Marvel and Teen Wolf\n",
    "marvel_bridge['fandom_group'] = 'Marvel'\n",
    "teen_wolf_bridge['fandom_group'] = 'Teen Wolf'\n",
    "marvel_tw_bridge_df = pd.concat([marvel_bridge, teen_wolf_bridge], ignore_index=True)\n",
    "\n",
    "# Save to CSV for easy export and further deep-dive analysis\n",
    "marvel_tw_bridge_df.to_csv('marvel_tw_bridge_works.csv', index=False)\n",
    "print(\"Saved filtered bridge works to 'marvel_tw_bridge_works.csv'.\")\n",
    "\n",
    "# 6. Calculate overall baselines for TARGET_METRIC\n",
    "marvel_overall_mean = marvel_df[TARGET_METRIC].mean()\n",
    "marvel_overall_median = marvel_df[TARGET_METRIC].median()\n",
    "\n",
    "tw_overall_mean = teen_wolf_df[TARGET_METRIC].mean()\n",
    "tw_overall_median = teen_wolf_df[TARGET_METRIC].median()\n",
    "\n",
    "# 7. Calculate bridge author work statistics for TARGET_METRIC\n",
    "marvel_bridge_mean = marvel_bridge[TARGET_METRIC].mean()\n",
    "marvel_bridge_median = marvel_bridge[TARGET_METRIC].median()\n",
    "marvel_bridge_ratio = marvel_bridge_median / marvel_overall_median if marvel_overall_median > 0 else 0\n",
    "\n",
    "tw_bridge_mean = teen_wolf_bridge[TARGET_METRIC].mean()\n",
    "tw_bridge_median = teen_wolf_bridge[TARGET_METRIC].median()\n",
    "tw_bridge_ratio = tw_bridge_median / tw_overall_median if tw_overall_median > 0 else 0\n",
    "\n",
    "# 8. Print summary statistics\n",
    "print(\"=\" * 60)\n",
    "print(f\"DEEP DIVE: MARVEL VS. TEEN WOLF (Metric: {TARGET_METRIC})\")\n",
    "print(\"=\" * 60)\n",
    "print(f\"Number of bridge authors identified: {len(bridge_authors)}\")\n",
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
    "# 9. Show a preview of the bridge works\n",
    "print(\"\\nPreview of Marvel vs. Teen Wolf Bridge Works:\")\n",
    "print(marvel_tw_bridge_df[['Title', 'author_str', 'fandom_group', TARGET_METRIC, 'Words']].head(15))\n",
    "\n",
    "# 10. Render a premium comparison visualization\n",
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
    "plt.show()"
]

# Find the cell index and replace
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        markdown_str = "".join(cell["source"])
        if "Deep Dive: Marvel (Mega-Fandom) vs. Teen Wolf" in markdown_str:
            cell["source"] = markdown_source
            found_markdown = True
            
            # The next cell should be the code cell
            if idx + 1 < len(nb["cells"]) and nb["cells"][idx+1]["cell_type"] == "code":
                nb["cells"][idx+1]["source"] = code_source
                nb["cells"][idx+1]["execution_count"] = None
                nb["cells"][idx+1]["outputs"] = []
                found_code = True

if found_markdown and found_code:
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Successfully updated cells in popularity_analysis.ipynb with author_str resolution!")
else:
    print(f"Error: markdown found={found_markdown}, code found={found_code}")
