import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AO3Visualizer:
    """
    A class to create modern, beautiful visualizations of the AO3 dataset,
    focusing on counts and distributions of hits/kudos grouped by other columns.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the visualizer with a copy of the dataframe and sets the aesthetic theme.
        """
        self.df = df.copy()
        
        # Set premium aesthetic styling parameters for matplotlib and seaborn
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
            'figure.titlesize': 15,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 150
        })

    def plot_work_counts(self, group_col='fandom_tag_1', top_n=10, palette='viridis', figsize=(10, 6)):
        """
        Plots a horizontal bar chart showing the number of fanfictions by group (e.g. fandom or author).
        """
        df_temp = self.df.copy()
        
        # Resolve group column if it is 'Authors'
        if group_col == 'Authors' and 'parsed_authors' in df_temp.columns:
            df_temp['author_str'] = df_temp['parsed_authors'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
            group_col_resolved = 'author_str'
            title_group = 'Author'
        else:
            group_col_resolved = group_col
            title_group = group_col.replace('_', ' ').title()

        if group_col_resolved not in df_temp.columns:
            raise ValueError(f"Grouping column '{group_col}' not found in DataFrame.")

        # Compute counts
        counts = df_temp[group_col_resolved].value_counts().head(top_n)
        
        plt.figure(figsize=figsize)
        ax = sns.barplot(x=counts.values, y=counts.index, palette=palette, hue=counts.index, legend=False)
        
        # Annotate bar values on the right of the bar
        for i, val in enumerate(counts.values):
            ax.text(val + (max(counts.values) * 0.01), i, f" {val:,}", va='center', fontsize=9, fontweight='bold')
            
        plt.title(f"Top {top_n} {title_group}s by Number of Fanfictions", pad=20, weight='bold')
        plt.xlabel("Number of Fanfictions", labelpad=10)
        plt.ylabel(title_group, labelpad=10)
        plt.tight_layout()
        plt.show()

    def plot_metric_distribution(self, metric_col='Hits', group_col='fandom_tag_1', top_n=5, log_scale=True, palette='coolwarm', figsize=(12, 7)):
        """
        Plots the distribution (using boxplots with strip overlay) of a metric (Hits, Kudos, Words, etc.)
        for the top N groups in the grouping column.
        """
        if metric_col not in self.df.columns:
            raise ValueError(f"Metric column '{metric_col}' not found in DataFrame.")
            
        df_temp = self.df.copy()
        
        # Resolve group column if it is 'Authors'
        if group_col == 'Authors' and 'parsed_authors' in df_temp.columns:
            df_temp['author_str'] = df_temp['parsed_authors'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
            group_col_resolved = 'author_str'
            title_group = 'Author'
        else:
            group_col_resolved = group_col
            title_group = group_col.replace('_', ' ').title()

        if group_col_resolved not in df_temp.columns:
            raise ValueError(f"Grouping column '{group_col}' not found in DataFrame.")

        # Get top N groups by frequency to focus on the largest categories
        top_groups = df_temp[group_col_resolved].value_counts().head(top_n).index
        df_filtered = df_temp[df_temp[group_col_resolved].isin(top_groups)].copy()

        # Handle log scaling for skewed distributions
        if log_scale:
            df_filtered[f'log_{metric_col}'] = np.log10(df_filtered[metric_col] + 1)
            y_col = f'log_{metric_col}'
            ylabel_suffix = " (Log10 Scale)"
        else:
            y_col = metric_col
            ylabel_suffix = ""

        plt.figure(figsize=figsize)
        
        # Draw boxplots
        sns.boxplot(
            data=df_filtered, 
            x=group_col_resolved, 
            y=y_col, 
            order=top_groups, 
            palette=palette,
            hue=group_col_resolved,
            legend=False,
            width=0.5,
            showfliers=not log_scale # Hide outlier markers if strip plot is overlayed on log scale
        )
        
        # Overlay points representing actual density
        if log_scale:
            sns.stripplot(
                data=df_filtered, 
                x=group_col_resolved, 
                y=y_col, 
                order=top_groups, 
                color='black', 
                alpha=0.1, 
                size=2.5, 
                jitter=0.25
            )

        plt.title(f"Distribution of {metric_col} across Top {top_n} {title_group}s", pad=20, weight='bold')
        plt.xlabel(title_group, labelpad=10)
        plt.ylabel(f"{metric_col}{ylabel_suffix}", labelpad=10)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_metric_summary(self, metric_col='Kudos', group_col='fandom_tag_1', top_n=10, estimator='median', palette='magma', figsize=(10, 6)):
        """
        Plots a summary bar chart comparing average/median of a metric across top groups.
        """
        if metric_col not in self.df.columns:
            raise ValueError(f"Metric column '{metric_col}' not found in DataFrame.")
            
        df_temp = self.df.copy()
        
        if group_col == 'Authors' and 'parsed_authors' in df_temp.columns:
            df_temp['author_str'] = df_temp['parsed_authors'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
            group_col_resolved = 'author_str'
            title_group = 'Author'
        else:
            group_col_resolved = group_col
            title_group = group_col.replace('_', ' ').title()

        if group_col_resolved not in df_temp.columns:
            raise ValueError(f"Grouping column '{group_col}' not found in DataFrame.")

        # Get top groups
        top_groups = df_temp[group_col_resolved].value_counts().head(top_n).index
        df_filtered = df_temp[df_temp[group_col_resolved].isin(top_groups)]

        # Calculate statistics
        if estimator == 'median':
            summary = df_filtered.groupby(group_col_resolved)[metric_col].median().loc[top_groups]
        else:
            summary = df_filtered.groupby(group_col_resolved)[metric_col].mean().loc[top_groups]

        plt.figure(figsize=figsize)
        ax = sns.barplot(x=summary.values, y=summary.index, palette=palette, hue=summary.index, legend=False)

        # Label actual values on the right
        for i, val in enumerate(summary.values):
            ax.text(val + (max(summary.values) * 0.01), i, f" {val:,.1f}", va='center', fontsize=9, fontweight='bold')

        plt.title(f"{estimator.title()} {metric_col} for Top {top_n} {title_group}s", pad=20, weight='bold')
        plt.xlabel(f"{estimator.title()} {metric_col}", labelpad=10)
        plt.ylabel(title_group, labelpad=10)
        plt.tight_layout()
        plt.show()

    def plot_fandoms_per_author(self, palette='Blues_r', figsize=(8, 5)):
        """
        Plots a bar chart showing the percentage of authors who have written works
        in 1, 2, 3, or 4+ unique fandoms (using fandom_tag_1).
        """
        df_temp = self.df.copy()
        
        # Ensure author_str and fandom_tag_1 are available
        if 'author_str' not in df_temp.columns:
            if 'parsed_authors' in df_temp.columns:
                df_temp['author_str'] = df_temp['parsed_authors'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
            else:
                df_temp['author_str'] = df_temp['Authors']

        if 'fandom_tag_1' not in df_temp.columns:
            if 'parsed_fandoms' in df_temp.columns:
                df_temp['fandom_tag_1'] = df_temp['parsed_fandoms'].apply(lambda x: x[0] if len(x) > 0 else None)
            else:
                df_temp['fandom_tag_1'] = df_temp['Fandom Tags']

        # Group by author and count unique fandoms
        author_fandom_counts = df_temp.groupby('author_str')['fandom_tag_1'].nunique()
        
        # Bin the counts
        def bin_fandoms(c):
            if c == 1: return '1 Fandom'
            elif c == 2: return '2 Fandoms'
            elif c == 3: return '3 Fandoms'
            else: return '4+ Fandoms'
            
        binned = author_fandom_counts.apply(bin_fandoms)
        order = ['1 Fandom', '2 Fandoms', '3 Fandoms', '4+ Fandoms']
        
        # Calculate percentages
        counts = binned.value_counts().reindex(order).fillna(0)
        percentages = (counts / counts.sum()) * 100
        
        plt.figure(figsize=figsize)
        colors = sns.color_palette(palette, len(order))
        ax = sns.barplot(x=percentages.index, y=percentages.values, palette=colors, hue=percentages.index, legend=False)
        
        # Annotate percentages on top of bars
        for i, val in enumerate(percentages.values):
            ax.text(i, val + 1, f"{val:.2f}%", ha='center', fontsize=10, fontweight='bold')
            
        plt.title("Distribution of Authors by Number of Fandoms", pad=20, weight='bold')
        plt.xlabel("Number of Fandoms Written In (using primary fandom)", labelpad=10)
        plt.ylabel("Percentage of Authors (%)", labelpad=10)
        plt.ylim(0, max(percentages.values) + 10)
        plt.tight_layout()
        plt.show()

    def plot_fics_per_author(self, palette='Purples_r', figsize=(8, 5)):
        """
        Plots a bar chart showing the percentage of authors who have written
        1, 2, 3 to 5, 6 to 10, or 11+ fanfictions in the dataset.
        """
        df_temp = self.df.copy()
        
        if 'author_str' not in df_temp.columns:
            if 'parsed_authors' in df_temp.columns:
                df_temp['author_str'] = df_temp['parsed_authors'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
            else:
                df_temp['author_str'] = df_temp['Authors']

        # Group by author and count works
        author_fic_counts = df_temp.groupby('author_str').size()
        
        # Bin the counts
        def bin_fics(c):
            if c == 1: return '1 Fic'
            elif c == 2: return '2 Fics'
            elif 3 <= c <= 5: return '3 to 5 Fics'
            elif 6 <= c <= 10: return '6 to 10 Fics'
            else: return '11+ Fics'
            
        binned = author_fic_counts.apply(bin_fics)
        order = ['1 Fic', '2 Fics', '3 to 5 Fics', '6 to 10 Fics', '11+ Fics']
        
        # Calculate percentages
        counts = binned.value_counts().reindex(order).fillna(0)
        percentages = (counts / counts.sum()) * 100
        
        plt.figure(figsize=figsize)
        colors = sns.color_palette(palette, len(order))
        ax = sns.barplot(x=percentages.index, y=percentages.values, palette=colors, hue=percentages.index, legend=False)
        
        # Annotate percentages on top of bars
        for i, val in enumerate(percentages.values):
            ax.text(i, val + 1, f"{val:.2f}%", ha='center', fontsize=10, fontweight='bold')
            
        plt.title("Distribution of Authors by Number of Fanfictions", pad=20, weight='bold')
        plt.xlabel("Number of Fanfictions Written", labelpad=10)
        plt.ylabel("Percentage of Authors (%)", labelpad=10)
        plt.ylim(0, max(percentages.values) + 10)
        plt.tight_layout()
        plt.show()

    def plot_cluster_comparison(self, df_clustered, metric_col='Hits', top_n=10, log_scale=True):
        """
        Plots side-by-side comparisons of:
        1. Fanfiction counts by original fandom_tag_1 vs. fandom_cluster.
        2. Distribution of a metric (e.g., Hits or Kudos) by original fandom_tag_1 vs. fandom_cluster.
        """
        if metric_col not in df_clustered.columns:
            raise ValueError(f"Metric column '{metric_col}' not found in DataFrame.")

        # Create figure for count comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Left panel: Top N original fandoms count
        orig_counts = df_clustered['fandom_tag_1'].value_counts().head(top_n)
        sns.barplot(x=orig_counts.values, y=orig_counts.index, palette='viridis', hue=orig_counts.index, legend=False, ax=axes[0])
        axes[0].set_title(f"Top {top_n} Original Fandoms by Fic Count", weight='bold', pad=10)
        axes[0].set_xlabel("Number of Fanfictions")
        axes[0].set_ylabel("Original Fandom Tag")
        for i, val in enumerate(orig_counts.values):
            axes[0].text(val + (max(orig_counts.values) * 0.01), i, f" {val:,}", va='center', fontsize=9, fontweight='bold')

        # 2. Right panel: Top N fandom clusters count
        clust_counts = df_clustered['fandom_cluster'].value_counts().head(top_n)
        sns.barplot(x=clust_counts.values, y=clust_counts.index, palette='plasma', hue=clust_counts.index, legend=False, ax=axes[1])
        axes[1].set_title(f"Top {top_n} Fandom Clusters by Fic Count", weight='bold', pad=10)
        axes[1].set_xlabel("Number of Fanfictions")
        axes[1].set_ylabel("Fandom Cluster")
        for i, val in enumerate(clust_counts.values):
            axes[1].text(val + (max(clust_counts.values) * 0.01), i, f" {val:,}", va='center', fontsize=9, fontweight='bold')

        plt.suptitle("Fic Count Comparison: Original Fandom Tags vs. Fandom Clusters", weight='bold', size=15, y=1.02)
        plt.tight_layout()
        plt.show()

        # Create figure for metric distribution comparison
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Extract top groups
        top_orig_groups = orig_counts.index
        top_clust_groups = clust_counts.index
        
        df_orig_filtered = df_clustered[df_clustered['fandom_tag_1'].isin(top_orig_groups)].copy()
        df_clust_filtered = df_clustered[df_clustered['fandom_cluster'].isin(top_clust_groups)].copy()

        if log_scale:
            df_orig_filtered[f'log_{metric_col}'] = np.log10(df_orig_filtered[metric_col] + 1)
            df_clust_filtered[f'log_{metric_col}'] = np.log10(df_clust_filtered[metric_col] + 1)
            y_col = f'log_{metric_col}'
            ylabel_suffix = " (Log10 Scale)"
        else:
            y_col = metric_col
            ylabel_suffix = ""

        # Top Panel: Original Fandoms Metric Distribution
        sns.boxplot(
            data=df_orig_filtered,
            x='fandom_tag_1',
            y=y_col,
            order=top_orig_groups,
            palette='viridis',
            hue='fandom_tag_1',
            legend=False,
            width=0.4,
            showfliers=not log_scale,
            ax=axes[0]
        )
        if log_scale:
            sns.stripplot(
                data=df_orig_filtered,
                x='fandom_tag_1',
                y=y_col,
                order=top_orig_groups,
                color='black',
                alpha=0.08,
                size=2,
                jitter=0.2,
                ax=axes[0]
            )
        axes[0].set_title(f"Distribution of {metric_col} across Top {top_n} Original Fandoms", weight='bold', pad=10)
        axes[0].set_xlabel("Original Fandom Tag")
        axes[0].set_ylabel(f"{metric_col}{ylabel_suffix}")
        axes[0].tick_params(axis='x', rotation=15)

        # Bottom Panel: Fandom Clusters Metric Distribution
        sns.boxplot(
            data=df_clust_filtered,
            x='fandom_cluster',
            y=y_col,
            order=top_clust_groups,
            palette='plasma',
            hue='fandom_cluster',
            legend=False,
            width=0.4,
            showfliers=not log_scale,
            ax=axes[1]
        )
        if log_scale:
            sns.stripplot(
                data=df_clust_filtered,
                x='fandom_cluster',
                y=y_col,
                order=top_clust_groups,
                color='black',
                alpha=0.08,
                size=2,
                jitter=0.2,
                ax=axes[1]
            )
        axes[1].set_title(f"Distribution of {metric_col} across Top {top_n} Fandom Clusters", weight='bold', pad=10)
        axes[1].set_xlabel("Fandom Cluster")
        axes[1].set_ylabel(f"{metric_col}{ylabel_suffix}")
        axes[1].tick_params(axis='x', rotation=15)

        plt.suptitle(f"{metric_col} Distribution Comparison: Original Fandom Tags vs. Fandom Clusters", weight='bold', size=15, y=1.02)
        plt.tight_layout()
        plt.show()
