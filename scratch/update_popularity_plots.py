import json
from pathlib import Path

notebook_path = Path("/Users/ele/ao3-project2/popularity_analysis.ipynb")
with open(notebook_path, "r") as f:
    nb = json.load(f)

modified = False

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        
        # Check for Cell 6 (correlation scatter plot)
        if "Plot Popularity Ratio vs Fandom Size" in source_str and "popularity_vs_fandom_size.png" not in source_str:
            # Insert plt.savefig before plt.show()
            new_source = []
            for line in cell["source"]:
                if "plt.show()" in line:
                    new_source.append("plt.savefig('docs/popularity_vs_fandom_size.png', dpi=300, bbox_inches='tight')\n")
                new_source.append(line)
            cell["source"] = new_source
            modified = True
            print("Added savefig to Cell 6 (scatter plot).")
            
        # Check for Cell 7 (deep dive comparison chart)
        if "Deep Dive: Bridge Authors in Marvel vs. Teen Wolf" in source_str and "marvel_teen_wolf_deep_dive.png" not in source_str:
            # Insert plt.savefig before plt.show()
            new_source = []
            for line in cell["source"]:
                if "plt.show()" in line:
                    new_source.append("plt.savefig('docs/marvel_teen_wolf_deep_dive.png', dpi=300, bbox_inches='tight')\n")
                new_source.append(line)
            cell["source"] = new_source
            modified = True
            print("Added savefig to Cell 7 (deep dive comparison chart).")

if modified:
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Notebook popularity_analysis.ipynb updated successfully with savefig outputs!")
else:
    print("No changes needed or notebook already updated.")
