import pandas as pd
import numpy as np
import re
from ao3_loader import AO3DatasetLoader
from ao3_manipulator import AO3DataManipulator

# Load data
loader = AO3DatasetLoader()
df = loader.get_preprocessed_dataframe()

# Clean data as in popularity_analysis.ipynb but include the <= 5 fandom tags filter
df_filtered = df[df['parsed_authors'].apply(lambda x: len(x) > 0 and x != ['orphan_account'])]
df_filtered = df_filtered[df_filtered['parsed_authors'].apply(lambda x: len(x) == 1)]
df_filtered = df_filtered[df_filtered['parsed_fandoms'].apply(lambda x: len(x) <= 5)] # Correct Filter 2: <= 5 fandom tags
final_df = df_filtered[df_filtered['Words'] > 0]

# Add first fandom tag and other features using manipulator
manipulator = AO3DataManipulator(final_df)
manipulated_df = manipulator.run_all_manipulations()
manipulated_df['author_str'] = manipulated_df['parsed_authors'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')

# Define Marvel and Teen Wolf filters based on Fandom Tags
marvel_pat = re.compile(r'marvel|avengers|mcu', re.IGNORECASE)
teen_wolf_pat = re.compile(r'teen wolf', re.IGNORECASE)

# Create boolean masks on Fandom Tags column
is_marvel = manipulated_df['Fandom Tags'].apply(lambda x: bool(marvel_pat.search(str(x))))
is_teen_wolf = manipulated_df['Fandom Tags'].apply(lambda x: bool(teen_wolf_pat.search(str(x))))

# Check crossover works (works tagged with both Marvel AND Teen Wolf)
is_crossover = is_marvel & is_teen_wolf
print(f"Number of crossover works between Marvel and Teen Wolf: {is_crossover.sum()}")

# Filter to get non-crossover works
marvel_only_df = manipulated_df[is_marvel & ~is_crossover].copy()
teen_wolf_only_df = manipulated_df[is_teen_wolf & ~is_crossover].copy()

# Find true bridge authors
marvel_only_authors = set(marvel_only_df['author_str'].unique())
teen_wolf_only_authors = set(teen_wolf_only_df['author_str'].unique())

true_bridge_authors = marvel_only_authors.intersection(teen_wolf_only_authors)
true_bridge_authors.discard('Unknown')

print(f"Number of true bridge authors: {len(true_bridge_authors)}")
print(f"True bridge authors list: {sorted(list(true_bridge_authors))}")

# Let's check their works in the two groups
marvel_bridge_works = marvel_only_df[marvel_only_df['author_str'].isin(true_bridge_authors)].copy()
teen_wolf_bridge_works = teen_wolf_only_df[teen_wolf_only_df['author_str'].isin(true_bridge_authors)].copy()

print(f"Marvel non-crossover bridge works count: {len(marvel_bridge_works)}")
print(f"Teen Wolf non-crossover bridge works count: {len(teen_wolf_bridge_works)}")

# Print Hits statistics (default TARGET_METRIC)
target_metric = 'Hits'

# Overall Marvel baselines (using non-crossover Marvel works or all Marvel works? Let's compute for all Marvel works)
marvel_median = manipulated_df[is_marvel][target_metric].median()
marvel_mean = manipulated_df[is_marvel][target_metric].mean()

# Overall Teen Wolf baselines (using all Teen Wolf works)
tw_median = manipulated_df[is_teen_wolf][target_metric].median()
tw_mean = manipulated_df[is_teen_wolf][target_metric].mean()

# Bridge authors' works stats
marvel_bridge_median = marvel_bridge_works[target_metric].median()
marvel_bridge_mean = marvel_bridge_works[target_metric].mean()
marvel_bridge_ratio = marvel_bridge_median / marvel_median if marvel_median > 0 else np.nan

tw_bridge_median = teen_wolf_bridge_works[target_metric].median()
tw_bridge_mean = teen_wolf_bridge_works[target_metric].mean()
tw_bridge_ratio = tw_bridge_median / tw_median if tw_median > 0 else np.nan

print("\n--- BASELINES ---")
print(f"Marvel Overall Mean: {marvel_mean:.2f}, Median: {marvel_median:.2f}")
print(f"Teen Wolf Overall Mean: {tw_mean:.2f}, Median: {tw_median:.2f}")

print("\n--- TRUE BRIDGE AUTHORS (NON-CROSSOVER WORKS) ---")
print(f"Marvel Bridge Works Mean: {marvel_bridge_mean:.2f}, Median: {marvel_bridge_median:.2f}, Ratio vs Median: {marvel_bridge_ratio:.4f}")
print(f"Teen Wolf Bridge Works Mean: {tw_bridge_mean:.2f}, Median: {tw_bridge_median:.2f}, Ratio vs Median: {tw_bridge_ratio:.4f}")
