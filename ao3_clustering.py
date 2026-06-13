import os
import ast
import json
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

class AO3FandomClustering:
    """
    A class to calculate co-occurrence overlap indexes between fandom tags,
    cluster overlapping fandoms using network connected components, and
    visualize the tag network.
    """
    def __init__(self, df: pd.DataFrame, min_occurrences=50, threshold=0.5):
        """
        Initializes the clustering manager.
        
        Parameters:
        - df: The preprocessed pandas DataFrame.
        - min_occurrences: Minimum number of works a fandom tag must have to be considered.
        - threshold: The similarity threshold (overlap index) above which tags are connected.
        """
        self.df = df.copy()
        self.min_occurrences = min_occurrences
        self.threshold = threshold
        self.graph = nx.Graph()
        self.tag_to_cluster = {}
        self.cluster_sizes = {}
        self.occurrences = {}
        self.id_map = {}

    def build_graph(self, min_co_occurrences=3):
        """
        Builds the networkx graph using the co-occurrence overlap index.
        For each pair of tags, the overlap index is computed as:
        S(A, B) = Co-Occurrences(A, B) / min(Occurrences(A), Occurrences(B))
        """
        print(f"Building fandom network graph on all unique tags (threshold={self.threshold}, min_co_occurrences={min_co_occurrences})...")
        
        # Ensure parsed fandoms column exists
        if 'parsed_fandoms' not in self.df.columns:
            def parse_list(val):
                if pd.isna(val): return []
                if isinstance(val, str):
                    try: return ast.literal_eval(val)
                    except: return [item.strip() for item in val.strip('[]').replace("'", "").split(',') if item.strip()]
                return []
            self.df['parsed_fandoms'] = self.df['Fandom Tags'].apply(parse_list)

        # Count occurrences of all fandom tags
        all_tags = []
        for fandoms in self.df['parsed_fandoms']:
            all_tags.extend(fandoms)
        
        tag_counts = pd.Series(all_tags).value_counts()
        self.occurrences = tag_counts.to_dict()
        valid_tags = list(tag_counts.index)
        print(f"Number of unique fandom tags: {len(valid_tags)}")

        # Create mapping from tag to set of row indices containing it for rapid co-occurrence/author lookup
        self.id_map = {tag: set() for tag in valid_tags}
        for idx, row in self.df.iterrows():
            for tag in row['parsed_fandoms']:
                if tag in self.id_map:
                    self.id_map[tag].add(idx)

        # Add all nodes to graph
        for tag in valid_tags:
            self.graph.add_node(tag, size=self.occurrences[tag])

        # Compute co-occurrences only for pairs that actually occur together
        from collections import defaultdict
        co_counts = defaultdict(int)
        for fandoms in self.df['parsed_fandoms']:
            unique_fandoms = list(set(fandoms))
            n = len(unique_fandoms)
            for i in range(n):
                for j in range(i + 1, n):
                    u, v = sorted([unique_fandoms[i], unique_fandoms[j]])
                    co_counts[(u, v)] += 1

        # Add edges meeting the overlap threshold
        edges_added = 0
        for (tag_a, tag_b), co_occurrences in co_counts.items():
            if co_occurrences >= min_co_occurrences:
                if tag_a in self.occurrences and tag_b in self.occurrences:
                    min_occ = min(self.occurrences[tag_a], self.occurrences[tag_b])
                    overlap_index = co_occurrences / min_occ
                    
                    if overlap_index >= self.threshold:
                        self.graph.add_edge(
                            tag_a, 
                            tag_b, 
                            weight=overlap_index, 
                            co_occurrences=co_occurrences,
                            distance=1.0 - overlap_index
                        )
                        edges_added += 1
                        
        print(f"Graph constructed with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        return self

    def create_clusters(self):
        """
        Finds connected components in the graph and groups tags into clusters.
        Each cluster is named after the most frequent tag in that component.
        """
        if self.graph.number_of_nodes() == 0:
            self.build_graph()

        print("Identifying fandom clusters via connected components...")
        components = list(nx.connected_components(self.graph))
        
        self.tag_to_cluster = {}
        cluster_info = []

        for i, comp in enumerate(components):
            # Find the tag with the highest frequency in the component to represent the cluster
            comp_list = list(comp)
            representative_tag = max(comp_list, key=lambda tag: self.occurrences[tag])
            
            # Map each tag in this component to the representative tag
            for tag in comp_list:
                self.tag_to_cluster[tag] = representative_tag
            
            # Track sizes of clusters of size > 1 (actual groups of merged fandoms)
            if len(comp) > 1:
                cluster_info.append((representative_tag, len(comp), [t for t in comp_list if t != representative_tag]))

        # Sort and print clusters
        cluster_info = sorted(cluster_info, key=lambda x: x[1], reverse=True)
        print(f"Created {len(cluster_info)} multi-tag clusters.")
        for idx, (rep, size, members) in enumerate(cluster_info[:10]):
            print(f" Cluster {idx+1}: '{rep}' (Size: {size}) -> Members: {members[:5]}{'...' if len(members) > 5 else ''}")
            
        return self

    def get_clustered_dataframe(self) -> pd.DataFrame:
        """
        Maps works' primary fandoms (fandom_tag_1) to their cluster names,
        and returns the modified DataFrame containing a 'fandom_cluster' column.
        """
        if not self.tag_to_cluster:
            self.create_clusters()

        df_clustered = self.df.copy()
        
        # Ensure fandom_tag_1 exists
        if 'fandom_tag_1' not in df_clustered.columns:
            if 'parsed_fandoms' in df_clustered.columns:
                df_clustered['fandom_tag_1'] = df_clustered['parsed_fandoms'].apply(lambda x: x[0] if len(x) > 0 else None)
            else:
                def parse_first(val):
                    if pd.isna(val): return None
                    try:
                        lst = ast.literal_eval(val)
                        return lst[0] if len(lst) > 0 else None
                    except:
                        return None
                df_clustered['fandom_tag_1'] = df_clustered['Fandom Tags'].apply(parse_first)

        # Map fandom_tag_1 to its cluster name, falling back to original if no cluster exists
        df_clustered['fandom_cluster'] = df_clustered['fandom_tag_1'].apply(
            lambda x: self.tag_to_cluster.get(x, x) if pd.notna(x) else x
        )
        return df_clustered

    def get_clustered_fandom_dataframe(self, threshold=0.7) -> pd.DataFrame:
        """
        Creates a new column 'clustered_fandom', which, for each fandom A which has an
        overlap index > threshold with B, is the biggest fandom between A and B (by work count).
        Also adds a binary column 'fandom_equals_clustered' (1 if fandom_tag_1 == clustered_fandom, else 0).
        """
        if self.graph.number_of_nodes() == 0:
            self.build_graph()

        # Build a temporary graph at the specified threshold (e.g. 0.7)
        edges_threshold = [(u, v) for u, v, d in self.graph.edges(data=True) if d['weight'] >= threshold]
        subgraph_threshold = nx.Graph()
        subgraph_threshold.add_nodes_from(self.graph.nodes())
        subgraph_threshold.add_edges_from(edges_threshold)
        
        # Get components at this threshold
        components = list(nx.connected_components(subgraph_threshold))
        
        tag_to_rep = {}
        for comp in components:
            comp_list = list(comp)
            # Find the largest fandom tag in this component (by occurrence count)
            rep_tag = max(comp_list, key=lambda tag: self.occurrences[tag])
            for tag in comp_list:
                tag_to_rep[tag] = rep_tag

        df_out = self.df.copy()
        
        # Ensure fandom_tag_1 exists
        if 'fandom_tag_1' not in df_out.columns:
            if 'parsed_fandoms' in df_out.columns:
                df_out['fandom_tag_1'] = df_out['parsed_fandoms'].apply(lambda x: x[0] if len(x) > 0 else None)
            else:
                def parse_first(val):
                    if pd.isna(val): return None
                    try:
                        lst = ast.literal_eval(val)
                        return lst[0] if len(lst) > 0 else None
                    except:
                        return None
                df_out['fandom_tag_1'] = df_out['Fandom Tags'].apply(parse_first)

        # Map to the clustered_fandom, falling back to original fandom_tag_1 if tag is below threshold/sparse
        df_out['clustered_fandom'] = df_out['fandom_tag_1'].apply(
            lambda x: tag_to_rep.get(x, x) if pd.notna(x) else x
        )
        
        # Binary column: 1 if fandom_tag_1 == clustered_fandom, else 0
        df_out['fandom_equals_clustered'] = (df_out['fandom_tag_1'] == df_out['clustered_fandom']).astype(int)
        
        return df_out

    def plot_fandom_network(self, figsize=(14, 10), min_cluster_size=2, min_occurrences=10):
        """
        Plots a network visualization of the fandom tags.
        Nodes are colored according to their connected component (cluster) to evidentiate grouping.
        Only displays connected components of size >= min_cluster_size to avoid clutter.
        """
        if self.graph.number_of_nodes() == 0:
            self.build_graph()
            
        # Filter nodes by minimum occurrences
        valid_nodes = [node for node in self.graph.nodes() if self.occurrences.get(node, 0) >= min_occurrences]
        subgraph_full = self.graph.subgraph(valid_nodes)
        
        # Filter the graph to only keep nodes in connected components of size >= min_cluster_size
        filtered_nodes = []
        for comp in nx.connected_components(subgraph_full):
            if len(comp) >= min_cluster_size:
                filtered_nodes.extend(comp)
                
        subgraph = subgraph_full.subgraph(filtered_nodes)
        
        if subgraph.number_of_nodes() == 0:
            print("No connected components found meeting the size/occurrence requirements.")
            return

        plt.figure(figsize=figsize)
        
        # Node positioning via spring layout (uses edge distance as forces)
        pos = nx.spring_layout(subgraph, weight='distance', k=0.15, seed=42)
        
        # Extract unique clusters in the subgraph for coloring
        subgraph_nodes = list(subgraph.nodes())
        groups = [self.tag_to_cluster.get(node, node) for node in subgraph_nodes]
        unique_groups = list(set(groups))
        
        # Color nodes dynamically based on cluster groups using HLS palette
        palette = sns.color_palette("hls", len(unique_groups))
        group_to_color = {g: palette[idx] for idx, g in enumerate(unique_groups)}
        node_colors = [group_to_color[g] for g in groups]
        
        # Calculate sizes and widths
        node_sizes = [self.occurrences[node] * 2 for node in subgraph.nodes()]
        edge_widths = [d['weight'] * 4 for u, v, d in subgraph.edges(data=True)]
        
        # Drawing nodes
        nx.draw_networkx_nodes(
            subgraph, 
            pos, 
            node_size=node_sizes, 
            node_color=node_colors, 
            alpha=0.8,
            linewidths=1.0,
            edgecolors='#1a1a1a'
        )
        
        # Drawing edges
        nx.draw_networkx_edges(
            subgraph, 
            pos, 
            width=edge_widths, 
            edge_color='#888888', 
            alpha=0.5
        )
        
        # Label nodes (with adjustments to prevent overlap)
        labels = {node: f"{node}\n({self.occurrences[node]})" for node in subgraph.nodes()}
        nx.draw_networkx_labels(
            subgraph, 
            pos, 
            labels=labels, 
            font_size=8, 
            font_family='sans-serif',
            font_weight='bold'
        )
        
        plt.title(f"Fandom Tag Network Graph (Colors Show Connected Clusters, Size >= {min_cluster_size})", pad=20, weight='bold', size=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def generate_interactive_network_html(self, filepath="fandom_network.html", min_cluster_size=2, show_bridges=True):
        """
        Generates a self-contained interactive, zoomable network visualization in HTML
        using Vis.js. Nodes are colored by cluster. Node sizes correspond to occurrence frequency,
        and edge widths correspond to the co-occurrence overlap index.
        Names are visualized only when zoomed in close enough to avoid crowding.
        Also adds dashed lavender edges across clusters to represent author crossovers if show_bridges is True.
        """
        if self.graph.number_of_nodes() == 0:
            self.build_graph()

        # Find connected nodes to avoid plotting isolated ones
        filtered_nodes = []
        for comp in nx.connected_components(self.graph):
            if len(comp) >= min_cluster_size:
                filtered_nodes.extend(comp)
                
        subgraph = self.graph.subgraph(filtered_nodes)
        
        if subgraph.number_of_nodes() == 0:
            print("No components found for interactive visualization.")
            return

        # Prepare unique clusters and assign colors
        subgraph_nodes = list(subgraph.nodes())
        groups = [self.tag_to_cluster.get(node, node) for node in subgraph_nodes]
        unique_groups = list(set(groups))
        
        # Color mapping (qualitative HLS color space in hex string formats)
        colors_rgb = sns.color_palette("hls", len(unique_groups))
        def rgb_to_hex(rgb):
            return '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        
        group_colors = {g: rgb_to_hex(colors_rgb[idx]) for idx, g in enumerate(unique_groups)}

        # Build nodes JSON array
        nodes_data = []
        for node in subgraph_nodes:
            rep = self.tag_to_cluster.get(node, node)
            color = group_colors[rep]
            size = self.occurrences[node]
            nodes_data.append({
                "id": node,
                "label": node,
                "value": int(size),
                "group": rep,
                "color": {
                    "background": color,
                    "border": "#1a1a1a",
                    "highlight": {
                        "background": color,
                        "border": "#121212"
                    }
                },
                "title": f"Fandom: {node}<br>Works count: {size:,}<br>Cluster Group: {rep}"
            })

        # Build edges JSON array with co-occurrence data
        edges_data = []
        for u, v, d in subgraph.edges(data=True):
            edges_data.append({
                "from": u,
                "to": v,
                "value": float(d['weight']),
                "title": f"Similarity S({u}, {v}) = {d['weight'] * 100:.1f}%<br>Co-occurrences = {d['co_occurrences']:,}",
                "color": {
                    "color": "#a0aec0",
                    "highlight": "#111111",
                    "opacity": 0.5
                }
            })

        # Add Author Crossover Bridge edges across clusters (dashed lavender lines)
        if show_bridges:
            df_temp = self.df.copy()
            if 'author_str' not in df_temp.columns:
                if 'parsed_authors' in df_temp.columns:
                    df_temp['author_str'] = df_temp['parsed_authors'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
                else:
                    df_temp['author_str'] = df_temp['Authors']
            df_temp = df_temp[df_temp['author_str'] != 'Unknown']

            tag_authors = {}
            for node in subgraph_nodes:
                if node in self.id_map:
                    row_indices = list(self.id_map[node])
                    valid_indices = [idx for idx in row_indices if idx in df_temp.index]
                    tag_authors[node] = set(df_temp.loc[valid_indices, 'author_str'])
                else:
                    tag_authors[node] = set()

            crossover_edges_added = 0
            for i in range(len(subgraph_nodes)):
                for j in range(i + 1, len(subgraph_nodes)):
                    u = subgraph_nodes[i]
                    v = subgraph_nodes[j]
                    
                    cluster_u = self.tag_to_cluster.get(u, u)
                    cluster_v = self.tag_to_cluster.get(v, v)
                    
                    if cluster_u != cluster_v:
                        common_authors = tag_authors[u] & tag_authors[v]
                        if len(common_authors) > 3:
                            edges_data.append({
                                "from": u,
                                "to": v,
                                "dashes": True,
                                "value": float(len(common_authors)),
                                "title": f"Author Crossover Bridge<br>Fandom A: {u} (Cluster: {cluster_u})<br>Fandom B: {v} (Cluster: {cluster_v})<br>Common Authors ({len(common_authors)}): {', '.join(sorted(list(common_authors))[:5])}{'...' if len(common_authors) > 5 else ''}",
                                "color": {
                                    "color": "#805ad5",
                                    "highlight": "#553c9a",
                                    "opacity": 0.6
                                }
                            })
                            crossover_edges_added += 1
            print(f"Added {crossover_edges_added} author crossover bridge edges across clusters to interactive network.")

        # Convert to JSON strings
        nodes_json = json.dumps(nodes_data)
        edges_json = json.dumps(edges_data)

        # Find the max occurrences to scale the slider dynamically
        max_node_occurrences = int(max(node["value"] for node in nodes_data)) if nodes_data else 100
        if max_node_occurrences < 10:
            max_node_occurrences = 100

        # HTML template using Vis.js with custom zoom label showing logic and styled glassmorphic tooltips
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AO3 Fandom Network Graph</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{
            background-color: #ffffff;
            color: #121212;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            overflow: hidden;
        }}
        #header {{
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 100;
            pointer-events: none;
        }}
        h1 {{
            margin: 0 0 5px 0;
            font-size: 24px;
            font-weight: 700;
            letter-spacing: 0.5px;
            color: #ff007b;
        }}
        p {{
            margin: 0;
            font-size: 13px;
            color: #555555;
        }}
        #mynetwork {{
            width: 100vw;
            height: 100vh;
            border: none;
            background-color: #ffffff;
        }}
        #instructions {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            z-index: 100;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 10px 15px;
            border-radius: 6px;
            border: 1px solid #ddd;
            font-size: 12px;
            color: #333;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            pointer-events: none;
        }}
        /* Beautiful light glassmorphic styling for tooltips */
        div.vis-tooltip {{
            position: absolute;
            visibility: hidden;
            background-color: rgba(255, 255, 255, 0.95) !important;
            border: 1px solid rgba(0, 0, 0, 0.15) !important;
            border-radius: 8px !important;
            color: #121212 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            font-size: 12px !important;
            padding: 8px 12px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
            backdrop-filter: blur(8px) !important;
            -webkit-backdrop-filter: blur(8px) !important;
            z-index: 1000 !important;
            pointer-events: none;
        }}
        /* Premium light search container styling */
        #search-container {{
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 100;
            width: 250px;
            background-color: rgba(255, 255, 255, 0.9);
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            padding: 8px;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
        }}
        #search-input {{
            width: calc(100% - 18px);
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 6px;
            outline: none;
            font-size: 13px;
            color: #121212;
            background-color: #ffffff;
        }}
        #search-input:focus {{
            border-color: #ff7b00;
        }}
        #search-results {{
            max-height: 200px;
            overflow-y: auto;
            margin-top: 5px;
            border-top: 1px solid #eee;
            display: none;
        }}
        .search-item {{
            padding: 8px;
            cursor: pointer;
            font-size: 12px;
            border-radius: 4px;
            color: #333;
            transition: background-color 0.2s;
        }}
        .search-item:hover {{
            background-color: #f7fafc;
            color: #000;
        }}
        /* Slower physics slider styling */
        #slider-container {{
            position: absolute;
            top: 120px;
            right: 20px;
            z-index: 100;
            width: 250px;
            background-color: rgba(255, 255, 255, 0.9);
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            padding: 10px;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            font-size: 13px;
        }}
        .slider-label {{
            font-weight: bold;
            color: #121212;
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }}
        .slider-input {{
            width: 100%;
            cursor: pointer;
        }}
        /* Glassmorphic loading screen */
        #loadingBar {{
            position: absolute;
            top: 0px;
            left: 0px;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.95);
            transition: all 0.5s ease;
            opacity: 1;
            z-index: 999;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }}
        .outerBorder {{
            width: 300px;
            height: 10px;
            border: 1px solid rgba(0, 0, 0, 0.1);
            background-color: #f3f3f3;
            border-radius: 5px;
            overflow: hidden;
            position: relative;
            margin-top: 15px;
        }}
        #text {{
            font-size: 16px;
            font-weight: 600;
            color: #ff7b00;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        #bar {{
            position: absolute;
            width: 0px;
            height: 100%;
            background: linear-gradient(45deg, #ff7b00, #ff007b);
            transition: width 0.1s ease;
        }}
        #loading-title {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 5px;
            color: #121212;
            letter-spacing: 0.5px;
        }}
    </style>
</head>
<body>
    <div id="loadingBar">
        <div id="loading-title">Stabilizing Network Layout...</div>
        <div id="text">0%</div>
        <div class="outerBorder">
            <div id="bar"></div>
        </div>
    </div>
    <div id="header">
        <h1>AO3 Fandom Network Graph</h1>
        <p>Interactive Co-Occurrence Network Visualization (Physics-Enabled)</p>
    </div>
    <div id="instructions">
        Scroll to zoom. Drag to pan. Zoom in closer to show fandom names. Hover over nodes/edges for details. Solid lines show co-occurrences.{" Dashed purple lines show author crossovers." if show_bridges else ""}
    </div>
    <div id="search-container">
        <input type="text" id="search-input" placeholder="Search fandom..." autocomplete="off">
        <div id="search-results"></div>
    </div>
    <div id="slider-container">
        <div class="slider-label">
            <span>Min Occurrences:</span>
            <span id="slider-val">10</span>
        </div>
        <input type="range" id="occ-slider" class="slider-input" min="1" max="{max_node_occurrences}" value="10">
    </div>
    <div id="mynetwork"></div>

    <script type="text/javascript">
        // Parse node and edge data from Python
        var originalNodes = {nodes_json};
        var edgesArray = {edges_json};

        // Convert string titles containing HTML line breaks into actual DOM elements for rendering
        originalNodes.forEach(function(node) {{
            if (node.title) {{
                var el = document.createElement("div");
                el.innerHTML = node.title;
                el.style.color = "#121212";
                el.style.fontSize = "12px";
                el.style.padding = "4px";
                node.title = el;
            }}
        }});
        edgesArray.forEach(function(edge) {{
            if (edge.title) {{
                var el = document.createElement("div");
                el.innerHTML = edge.title;
                el.style.color = "#121212";
                el.style.fontSize = "12px";
                el.style.padding = "4px";
                edge.title = el;
            }}
        }});

        // Create DataSet objects
        var nodesDataSet = new vis.DataSet(originalNodes);
        var edgesDataSet = new vis.DataSet(edgesArray);

        // Slider filtering value
        var currentMinOcc = 10;

        var nodesView = new vis.DataView(nodesDataSet, {{
            filter: function (node) {{
                return node.value >= currentMinOcc;
            }}
        }});

        // Container element
        var container = document.getElementById('mynetwork');

        // Graph data
        var data = {{
            nodes: nodesView,
            edges: edgesDataSet
        }};

        // Options configuration
        var options = {{
            nodes: {{
                shape: 'dot',
                font: {{
                    size: 13,
                    color: '#121212',
                    face: 'Segoe UI, Arial, sans-serif',
                    strokeWidth: 2,
                    strokeColor: '#ffffff'
                }},
                scaling: {{
                    min: 10,
                    max: 40
                }}
            }},
            edges: {{
                scaling: {{
                    min: 0.5,
                    max: 6
                }},
                smooth: {{
                    type: 'continuous'
                }}
            }},
            physics: {{
                maxVelocity: 2,
                timestep: 0.05,
                forceAtlas2Based: {{
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 100,
                    springConstant: 0.02,
                    damping: 0.6
                }},
                solver: 'forceAtlas2Based',
                stabilization: {{
                    iterations: 2000,
                    updateInterval: 100
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                hideEdgesOnDrag: true
            }}
        }};

        // Initialize Network
        var network = new vis.Network(container, data, options);

        // Handle stabilization progress
        var progressStarted = false;
        network.on("stabilizationProgress", function(params) {{
            progressStarted = true;
            var width = params.iterations / params.total;
            var bar = document.getElementById('bar');
            var text = document.getElementById('text');
            if (bar && text) {{
                bar.style.width = (width * 100) + '%';
                text.innerHTML = Math.round(width * 100) + '%';
            }}
        }});

        function hideLoadingScreen() {{
            var bar = document.getElementById('bar');
            var text = document.getElementById('text');
            var loadingBar = document.getElementById('loadingBar');
            if (bar) bar.style.width = '100%';
            if (text) text.innerHTML = '100%';
            if (loadingBar) {{
                loadingBar.style.opacity = '0';
                setTimeout(function () {{
                    loadingBar.style.display = 'none';
                }}, 500);
            }}
        }}

        network.on("stabilizationIterationsDone", function () {{
            hideLoadingScreen();
        }});

        network.on("stabilized", function (params) {{
            if (progressStarted || (params && params.iterations > 0)) {{
                hideLoadingScreen();
            }}
        }});

        // Zoom-sensitive labels logic: Hide labels when zoomed out, show when zoomed in
        var labelsVisible = true;
        var zoomThreshold = 0.65; // Threshold for label visibility

        network.on("zoom", function(params) {{
            var zoomScale = params.scale;
            if (zoomScale < zoomThreshold) {{
                if (labelsVisible) {{
                    var updatedNodes = originalNodes.map(function(node) {{
                        return {{ id: node.id, label: "" }};
                    }});
                    nodesDataSet.update(updatedNodes);
                    labelsVisible = false;
                }}
            }} else {{
                if (!labelsVisible) {{
                    var updatedNodes = originalNodes.map(function(node) {{
                        return {{ id: node.id, label: node.label }};
                    }});
                    nodesDataSet.update(updatedNodes);
                    labelsVisible = true;
                }}
            }}
        }});

        // Search functionality
        var searchInput = document.getElementById('search-input');
        var searchResults = document.getElementById('search-results');

        searchInput.addEventListener('input', function() {{
            var val = this.value.toLowerCase().trim();
            searchResults.innerHTML = '';
            if (!val) {{
                searchResults.style.display = 'none';
                return;
            }}
            var matches = originalNodes.filter(function(node) {{
                return node.label && node.label.toLowerCase().includes(val);
            }});
            if (matches.length === 0) {{
                searchResults.style.display = 'none';
                return;
            }}
            searchResults.style.display = 'block';
            matches.forEach(function(node) {{
                var div = document.createElement('div');
                div.className = 'search-item';
                div.innerText = node.label;
                div.addEventListener('click', function() {{
                    searchInput.value = node.label;
                    searchResults.style.display = 'none';
                    
                    // Automatically lower slider if node is hidden
                    if (node.value < currentMinOcc) {{
                        currentMinOcc = node.value;
                        document.getElementById('occ-slider').value = currentMinOcc;
                        document.getElementById('slider-val').innerText = currentMinOcc;
                        nodesView.refresh();
                    }}
                    
                    // Focus and zoom on the selected node
                    network.selectNodes([node.id]);
                    network.focus(node.id, {{
                        scale: 1.2,
                        animation: {{
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }}
                    }});
                }});
                searchResults.appendChild(div);
            }});
        }});

        // Close search results when clicking outside
        document.addEventListener('click', function(e) {{
            var container = document.getElementById('search-container');
            if (container && !container.contains(e.target)) {{
                searchResults.style.display = 'none';
            }}
        }});

        // Slider functionality
        var slider = document.getElementById('occ-slider');
        var sliderVal = document.getElementById('slider-val');

        slider.addEventListener('input', function() {{
            currentMinOcc = parseInt(this.value);
            sliderVal.innerText = currentMinOcc;
            nodesView.refresh();
        }});
    </script>
</body>
</html>
"""
        with open(filepath, "w") as f:
            f.write(html_template)
        print(f"Interactive Vis.js network visualization saved to: {filepath}")

    def get_bridge_authors(self, df_clustered, cluster_col=None) -> pd.Series:
        """
        Identifies authors who have written works in more than one fandom cluster.
        Returns a Series of bridge authors and the number of unique clusters they write in.
        """
        df_temp = df_clustered.copy()
        if 'author_str' not in df_temp.columns:
            if 'parsed_authors' in df_temp.columns:
                df_temp['author_str'] = df_temp['parsed_authors'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
            else:
                df_temp['author_str'] = df_temp['Authors']

        # Exclude default/unknown authors
        df_temp = df_temp[df_temp['author_str'] != 'Unknown']

        # Determine cluster column automatically if not specified
        if cluster_col is None:
            if 'fandom_cluster' in df_temp.columns:
                cluster_col = 'fandom_cluster'
            elif 'clustered_fandom' in df_temp.columns:
                cluster_col = 'clustered_fandom'
            else:
                raise KeyError("Neither 'fandom_cluster' nor 'clustered_fandom' found in the DataFrame. Please specify cluster_col.")

        author_cluster_counts = df_temp.groupby('author_str')[cluster_col].nunique()
        bridge_authors = author_cluster_counts[author_cluster_counts > 1].sort_values(ascending=False)
        return bridge_authors

    def compute_author_overlap(self, df_clustered, cluster_col=None) -> pd.DataFrame:
        """
        Calculates the author overlap index between all pairs of fandom clusters.
        For clusters X and Y, the index is:
        Overlap(X, Y) = |Authors(X) & Authors(Y)| / min(|Authors(X)|, |Authors(Y)|)
        
        Returns a DataFrame summarizing the overlaps and identifying bridge authors.
        """
        df_temp = df_clustered.copy()
        if 'author_str' not in df_temp.columns:
            if 'parsed_authors' in df_temp.columns:
                df_temp['author_str'] = df_temp['parsed_authors'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
            else:
                df_temp['author_str'] = df_temp['Authors']

        # Exclude default/unknown authors
        df_temp = df_temp[df_temp['author_str'] != 'Unknown']

        # Determine cluster column automatically if not specified
        if cluster_col is None:
            if 'fandom_cluster' in df_temp.columns:
                cluster_col = 'fandom_cluster'
            elif 'clustered_fandom' in df_temp.columns:
                cluster_col = 'clustered_fandom'
            else:
                raise KeyError("Neither 'fandom_cluster' nor 'clustered_fandom' found in the DataFrame. Please specify cluster_col.")

        # Get set of unique authors for each cluster, excluding 'Unknown'
        cluster_authors = df_temp.groupby(cluster_col)['author_str'].apply(set).to_dict()
        clusters = list(cluster_authors.keys())
        num_clusters = len(clusters)
        
        overlap_results = []
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                c_x = clusters[i]
                c_y = clusters[j]
                
                authors_x = cluster_authors[c_x]
                authors_y = cluster_authors[c_y]
                
                if len(authors_x) == 0 or len(authors_y) == 0:
                    continue
                    
                common_authors = authors_x & authors_y
                if len(common_authors) > 0:
                    min_authors = min(len(authors_x), len(authors_y))
                    index = len(common_authors) / min_authors
                    overlap_results.append({
                        "Cluster_A": c_x,
                        "Cluster_B": c_y,
                        "Authors_A_Count": len(authors_x),
                        "Authors_B_Count": len(authors_y),
                        "Common_Authors_Count": len(common_authors),
                        "Author_Overlap_Index": index,
                        "Bridge_Authors": sorted(list(common_authors))
                    })
                    
        df_overlap = pd.DataFrame(overlap_results)
        if not df_overlap.empty:
            df_overlap = df_overlap.sort_values(by="Author_Overlap_Index", ascending=False).reset_index(drop=True)
        else:
            df_overlap = pd.DataFrame(columns=["Cluster_A", "Cluster_B", "Authors_A_Count", "Authors_B_Count", "Common_Authors_Count", "Author_Overlap_Index", "Bridge_Authors"])
            
        return df_overlap

    def plot_cluster_author_network(self, df_clustered, cluster_col=None, figsize=(12, 8), min_overlap=0.01, min_occurrences=10):
        """
        Plots a static network visualization of the fandom clusters based on author overlap.
        Nodes are clusters, sized by the number of unique authors, and edges represent
        the author overlap index.
        """
        df_overlap = self.compute_author_overlap(df_clustered, cluster_col=cluster_col)
        
        # Determine cluster column automatically
        if cluster_col is None:
            if 'fandom_cluster' in df_clustered.columns:
                cluster_col = 'fandom_cluster'
            elif 'clustered_fandom' in df_clustered.columns:
                cluster_col = 'clustered_fandom'
            else:
                raise KeyError("Neither 'fandom_cluster' nor 'clustered_fandom' found.")
                
        # Exclude 'Unknown'
        df_temp = df_clustered.copy()
        if 'author_str' not in df_temp.columns:
            if 'parsed_authors' in df_temp.columns:
                df_temp['author_str'] = df_temp['parsed_authors'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
            else:
                df_temp['author_str'] = df_temp['Authors']
        df_temp = df_temp[df_temp['author_str'] != 'Unknown']
        
        # Get count of authors per cluster
        cluster_author_counts = df_temp.groupby(cluster_col)['author_str'].nunique().to_dict()
        
        # Filter by occurrences of the representative fandom
        cluster_author_filtered = {}
        for c, count in cluster_author_counts.items():
            occurrences_count = self.occurrences.get(c, 0)
            if occurrences_count == 0:
                members = [tag for tag, cl in self.tag_to_cluster.items() if cl == c]
                occurrences_count = sum(self.occurrences.get(m, 0) for m in members)
            if occurrences_count >= min_occurrences:
                cluster_author_filtered[c] = count
                
        # Build networkx graph
        g = nx.Graph()
        
        # Add nodes
        for cluster, count in cluster_author_filtered.items():
            g.add_node(cluster, size=count)
            
        # Add edges above threshold
        for _, row in df_overlap.iterrows():
            if row['Author_Overlap_Index'] >= min_overlap:
                if row['Cluster_A'] in cluster_author_filtered and row['Cluster_B'] in cluster_author_filtered:
                    g.add_edge(
                        row['Cluster_A'],
                        row['Cluster_B'],
                        weight=row['Author_Overlap_Index'],
                        common_count=row['Common_Authors_Count'],
                        distance=1.0 - row['Author_Overlap_Index']
                    )
                
        # Only plot connected components or nodes with edges to keep it clean
        connected_nodes = [node for node, degree in dict(g.degree()).items() if degree > 0]
        subg = g.subgraph(connected_nodes)
        
        if subg.number_of_nodes() == 0:
            print("No cluster overlaps found meeting the threshold.")
            return
            
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(subg, weight='distance', k=0.3, seed=42)
        
        # Node colors
        nodes_list = list(subg.nodes())
        palette = sns.color_palette("hls", len(nodes_list))
        node_colors = [palette[i] for i in range(len(nodes_list))]
        
        # Sizes and widths
        node_sizes = [subg.nodes[node]['size'] * 20 for node in subg.nodes()]
        edge_widths = [d['weight'] * 10 for u, v, d in subg.edges(data=True)]
        
        nx.draw_networkx_nodes(
            subg,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8,
            linewidths=1.0,
            edgecolors='#1a1a1a'
        )
        
        nx.draw_networkx_edges(
            subg,
            pos,
            width=edge_widths,
            edge_color='#888888',
            alpha=0.5
        )
        
        labels = {node: f"{node}\n({subg.nodes[node]['size']} authors)" for node in subg.nodes()}
        nx.draw_networkx_labels(
            subg,
            pos,
            labels=labels,
            font_size=9,
            font_family='sans-serif',
            font_weight='bold'
        )
        
        plt.title("Cluster Crossover Network (Edges show Author Overlap Index)", pad=20, weight='bold', size=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def generate_cluster_author_network_html(self, df_clustered, cluster_col=None, filepath="cluster_author_network.html", min_overlap=0.0):
        """
        Generates a self-contained interactive, zoomable Vis.js network visualization
        of the clusters and their author overlap crossover bridges.
        """
        df_overlap = self.compute_author_overlap(df_clustered, cluster_col=cluster_col)
        
        if cluster_col is None:
            if 'fandom_cluster' in df_clustered.columns:
                cluster_col = 'fandom_cluster'
            elif 'clustered_fandom' in df_clustered.columns:
                cluster_col = 'clustered_fandom'
            else:
                raise KeyError("Neither 'fandom_cluster' nor 'clustered_fandom' found.")
                
        # Exclude 'Unknown'
        df_temp = df_clustered.copy()
        if 'author_str' not in df_temp.columns:
            if 'parsed_authors' in df_temp.columns:
                df_temp['author_str'] = df_temp['parsed_authors'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
            else:
                df_temp['author_str'] = df_temp['Authors']
        df_temp = df_temp[df_temp['author_str'] != 'Unknown']
        
        # Get count of authors per cluster
        cluster_author_counts = df_temp.groupby(cluster_col)['author_str'].nunique().to_dict()
        
        # Gather member fandoms for tooltips
        cluster_members = {}
        for cluster in cluster_author_counts.keys():
            members = [tag for tag, cl in self.tag_to_cluster.items() if cl == cluster]
            if not members:
                members = [cluster]
            cluster_members[cluster] = members

        # Filter nodes that have at least one overlap or are valid clusters to keep it clean
        connected_clusters = set()
        for _, row in df_overlap.iterrows():
            if row['Author_Overlap_Index'] >= min_overlap:
                if row['Cluster_A'] in cluster_author_counts and row['Cluster_B'] in cluster_author_counts:
                    connected_clusters.add(row['Cluster_A'])
                    connected_clusters.add(row['Cluster_B'])
                
        if not connected_clusters:
            # Fallback to all clusters that exist in our dataset to make sure it doesn't crash
            connected_clusters = set(cluster_author_counts.keys())

        # Prepare colors
        unique_groups = list(connected_clusters)
        colors_rgb = sns.color_palette("hls", len(unique_groups))
        def rgb_to_hex(rgb):
            return '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        group_colors = {g: rgb_to_hex(colors_rgb[idx]) for idx, g in enumerate(unique_groups)}

        # Build nodes JSON
        nodes_data = []
        for cluster in connected_clusters:
            color = group_colors.get(cluster, "#90cdf4")
            size = cluster_author_counts.get(cluster, 0)
            members_str = ", ".join(cluster_members.get(cluster, [cluster])[:10]) + ("..." if len(cluster_members.get(cluster, [cluster])) > 10 else "")
            
            # The occurrences value is set to the occurrence count of the representative tag
            # to be used for the min occurrences filtering slider.
            occurrences_count = int(self.occurrences.get(cluster, 0))
            if occurrences_count == 0:
                # If it's a multi-tag cluster, sum the occurrences of members
                occurrences_count = sum(self.occurrences.get(m, 0) for m in cluster_members.get(cluster, []))
            
            nodes_data.append({
                "id": cluster,
                "label": f"{cluster} Cluster",
                "value": int(size),
                "occurrences": int(occurrences_count),
                "color": {
                    "background": color,
                    "border": "#1a1a1a",
                    "highlight": {
                        "background": color,
                        "border": "#121212"
                    }
                },
                "title": f"Cluster Representative: {cluster}<br>Unique Authors: {size:,}<br>Cluster Max Occurrences: {occurrences_count:,}<br>Fandom Tags in Cluster: {members_str}"
            })

        # Build edges JSON
        edges_data = []
        for _, row in df_overlap.iterrows():
            if row['Cluster_A'] in connected_clusters and row['Cluster_B'] in connected_clusters:
                idx_val = row['Author_Overlap_Index']
                if idx_val >= min_overlap:
                    size_A = cluster_author_counts.get(row['Cluster_A'], 0)
                    size_B = cluster_author_counts.get(row['Cluster_B'], 0)
                    edges_data.append({
                        "from": row['Cluster_A'],
                        "to": row['Cluster_B'],
                        "value": float(idx_val),
                        "author_count": int(row['Common_Authors_Count']),
                        "title": f"Cluster A: {row['Cluster_A']} (Size: {size_A:,})<br>Cluster B: {row['Cluster_B']} (Size: {size_B:,})<br>Common Authors: {row['Common_Authors_Count']:,}<br>Author Overlap Index: {idx_val * 100:.1f}%",
                        "color": {
                            "color": "#805ad5",
                            "highlight": "#553c9a",
                            "opacity": 0.6
                        }
                    })

        # Dynamically determine the maximum value for the sliders based on the data
        max_node_occurrences = int(max(node["occurrences"] for node in nodes_data)) if nodes_data else 100
        if max_node_occurrences < 10:
            max_node_occurrences = 100
            
        max_bridge_authors = int(df_overlap['Common_Authors_Count'].max()) if not df_overlap.empty else 50
        if max_bridge_authors < 1:
            max_bridge_authors = 50

        nodes_json = json.dumps(nodes_data)
        edges_json = json.dumps(edges_data)

        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AO3 Cluster Crossover Network</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{
            background-color: #ffffff;
            color: #121212;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            overflow: hidden;
        }}
        #header {{
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 100;
            pointer-events: none;
        }}
        h1 {{
            margin: 0 0 5px 0;
            font-size: 24px;
            font-weight: 700;
            letter-spacing: 0.5px;
            color: #6c5ce7;
        }}
        p {{
            margin: 0;
            font-size: 13px;
            color: #555555;
        }}
        #mynetwork {{
            width: 100vw;
            height: 100vh;
            border: none;
            background-color: #ffffff;
        }}
        #instructions {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            z-index: 100;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 10px 15px;
            border-radius: 6px;
            border: 1px solid #ddd;
            font-size: 12px;
            color: #333;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            pointer-events: none;
        }}
        div.vis-tooltip {{
            position: absolute;
            visibility: hidden;
            background-color: rgba(255, 255, 255, 0.95) !important;
            border: 1px solid rgba(0, 0, 0, 0.15) !important;
            border-radius: 8px !important;
            color: #121212 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            font-size: 12px !important;
            padding: 8px 12px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
            backdrop-filter: blur(8px) !important;
            -webkit-backdrop-filter: blur(8px) !important;
            z-index: 1000 !important;
            pointer-events: none;
        }}
        /* Premium light search container styling */
        #search-container {{
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 100;
            width: 250px;
            background-color: rgba(255, 255, 255, 0.9);
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            padding: 8px;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
        }}
        #search-input {{
            width: calc(100% - 18px);
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 6px;
            outline: none;
            font-size: 13px;
            color: #121212;
            background-color: #ffffff;
        }}
        #search-input:focus {{
            border-color: #a29bfe;
        }}
        #search-results {{
            max-height: 200px;
            overflow-y: auto;
            margin-top: 5px;
            border-top: 1px solid #eee;
            display: none;
        }}
        .search-item {{
            padding: 8px;
            cursor: pointer;
            font-size: 12px;
            border-radius: 4px;
            color: #333;
            transition: background-color 0.2s;
        }}
        .search-item:hover {{
            background-color: #f7fafc;
            color: #000;
        }}
        /* Slower physics slider styling */
        #slider-container {{
            position: absolute;
            top: 120px;
            right: 20px;
            z-index: 100;
            width: 250px;
            background-color: rgba(255, 255, 255, 0.9);
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            padding: 10px;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            font-size: 13px;
        }}
        .slider-label {{
            font-weight: bold;
            color: #121212;
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }}
        .slider-input {{
            width: 100%;
            cursor: pointer;
        }}
        /* Glassmorphic loading screen */
        #loadingBar {{
            position: absolute;
            top: 0px;
            left: 0px;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.95);
            transition: all 0.5s ease;
            opacity: 1;
            z-index: 999;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }}
        .outerBorder {{
            width: 300px;
            height: 10px;
            border: 1px solid rgba(0, 0, 0, 0.1);
            background-color: #f3f3f3;
            border-radius: 5px;
            overflow: hidden;
            position: relative;
            margin-top: 15px;
        }}
        #text {{
            font-size: 16px;
            font-weight: 600;
            color: #6c5ce7;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        #bar {{
            position: absolute;
            width: 0px;
            height: 100%;
            background: linear-gradient(45deg, #a29bfe, #6c5ce7);
            transition: width 0.1s ease;
        }}
        #loading-title {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 5px;
            color: #121212;
            letter-spacing: 0.5px;
        }}
        /* Button Group styling */
        .btn-group {{
            display: flex;
            background-color: #f1f3f5;
            border-radius: 6px;
            padding: 2px;
            border: 1px solid #ddd;
        }}
        .toggle-btn {{
            border: none;
            background: none;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 600;
            border-radius: 4px;
            cursor: pointer;
            color: #666;
            transition: all 0.2s;
        }}
        .toggle-btn.active {{
            background-color: #6c5ce7;
            color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div id="loadingBar">
        <div id="loading-title">Stabilizing Network Layout...</div>
        <div id="text">0%</div>
        <div class="outerBorder">
            <div id="bar"></div>
        </div>
    </div>
    <div id="header">
        <h1>Authors as bridges across fandoms</h1>
        <p>A dynamic view of how authors common across fandoms can act as bridges</p>
    </div>
    <div id="instructions">
        Scroll to zoom. Drag to pan. Hover over nodes (clusters) or edges (bridges) for details.
    </div>
    <div id="search-container">
        <input type="text" id="search-input" placeholder="Search cluster..." autocomplete="off">
        <div id="search-results"></div>
    </div>
    <div id="slider-container">
        <div class="slider-label">
            <span>Min Occurrences:</span>
            <span id="slider-val">10</span>
        </div>
        <input type="range" id="occ-slider" class="slider-input" min="1" max="{max_node_occurrences}" value="10">
        
        <div class="slider-label" style="margin-top: 12px;">
            <span>Min Bridge Authors:</span>
            <span id="bridge-slider-val">1</span>
        </div>
        <input type="range" id="bridge-slider" class="slider-input" min="1" max="{max_bridge_authors}" value="1">

        <div class="slider-label" style="margin-top: 12px; align-items: center;">
            <span>Show Isolated Clusters:</span>
            <div class="btn-group">
                <button type="button" id="btn-show-isolated-yes" class="toggle-btn active">Yes</button>
                <button type="button" id="btn-show-isolated-no" class="toggle-btn">No</button>
            </div>
        </div>
    </div>
    <div id="mynetwork"></div>

    <script type="text/javascript">
        var originalNodes = {nodes_json};
        var edgesArray = {edges_json};

        originalNodes.forEach(function(node) {{
            if (node.title) {{
                var el = document.createElement("div");
                el.innerHTML = node.title;
                el.style.color = "#121212";
                el.style.fontSize = "12px";
                el.style.padding = "4px";
                node.title = el;
            }}
        }});
        edgesArray.forEach(function(edge) {{
            if (edge.title) {{
                var el = document.createElement("div");
                el.innerHTML = edge.title;
                el.style.color = "#121212";
                el.style.fontSize = "12px";
                el.style.padding = "4px";
                edge.title = el;
            }}
        }});

        var nodesDataSet = new vis.DataSet(originalNodes);
        var edgesDataSet = new vis.DataSet(edgesArray);

        // Slider filtering values
        var currentMinOcc = 10;
        var currentMinAuthors = 1;
        var showIsolated = true;

        var edgesView = new vis.DataView(edgesDataSet, {{
            filter: function (edge) {{
                return !edge.hasOwnProperty('author_count') || edge.author_count >= currentMinAuthors;
            }}
        }});

        var nodesView = new vis.DataView(nodesDataSet, {{
            filter: function (node) {{
                if (node.occurrences < currentMinOcc) {{
                    return false;
                }}
                if (!showIsolated) {{
                    var activeEdges = edgesView.get({{
                        filter: function(edge) {{
                            return edge.from === node.id || edge.to === node.id;
                        }}
                    }});
                    if (activeEdges.length === 0) {{
                        return false;
                    }}
                }}
                return true;
            }}
        }});

        var container = document.getElementById('mynetwork');

        var data = {{
            nodes: nodesView,
            edges: edgesView
        }};

        var options = {{
            nodes: {{
                shape: 'dot',
                font: {{
                    size: 14,
                    color: '#121212',
                    face: 'Segoe UI, Arial, sans-serif',
                    strokeWidth: 2,
                    strokeColor: '#ffffff'
                }},
                scaling: {{
                    min: 15,
                    max: 45
                }}
            }},
            edges: {{
                scaling: {{
                    min: 1,
                    max: 8
                }},
                smooth: {{
                    type: 'continuous'
                }}
            }},
            physics: {{
                maxVelocity: 2,
                timestep: 0.05,
                forceAtlas2Based: {{
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 150,
                    springConstant: 0.02,
                    damping: 0.6
                }},
                solver: 'forceAtlas2Based',
                stabilization: {{
                    iterations: 2000,
                    updateInterval: 100
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200
            }}
        }};

        var network = new vis.Network(container, data, options);

        // Handle stabilization progress
        var progressStarted = false;
        network.on("stabilizationProgress", function(params) {{
            progressStarted = true;
            var width = params.iterations / params.total;
            var bar = document.getElementById('bar');
            var text = document.getElementById('text');
            if (bar && text) {{
                bar.style.width = (width * 100) + '%';
                text.innerHTML = Math.round(width * 100) + '%';
            }}
        }});

        function hideLoadingScreen() {{
            var bar = document.getElementById('bar');
            var text = document.getElementById('text');
            var loadingBar = document.getElementById('loadingBar');
            if (bar) bar.style.width = '100%';
            if (text) text.innerHTML = '100%';
            if (loadingBar) {{
                loadingBar.style.opacity = '0';
                setTimeout(function () {{
                    loadingBar.style.display = 'none';
                }}, 500);
            }}
        }}

        network.on("stabilizationIterationsDone", function () {{
            hideLoadingScreen();
        }});

        network.on("stabilized", function (params) {{
            if (progressStarted || (params && params.iterations > 0)) {{
                hideLoadingScreen();
            }}
        }});

        // Search functionality
        var searchInput = document.getElementById('search-input');
        var searchResults = document.getElementById('search-results');

        searchInput.addEventListener('input', function() {{
            var val = this.value.toLowerCase().trim();
            searchResults.innerHTML = '';
            if (!val) {{
                searchResults.style.display = 'none';
                return;
            }}
            var matches = originalNodes.filter(function(node) {{
                return node.label && node.label.toLowerCase().includes(val);
            }});
            if (matches.length === 0) {{
                searchResults.style.display = 'none';
                return;
            }}
            searchResults.style.display = 'block';
            matches.forEach(function(node) {{
                var div = document.createElement('div');
                div.className = 'search-item';
                div.innerText = node.label;
                div.addEventListener('click', function() {{
                    searchInput.value = node.label;
                    searchResults.style.display = 'none';
                    
                    // Automatically lower slider if node is hidden
                    if (node.occurrences < currentMinOcc) {{
                        currentMinOcc = node.occurrences;
                        document.getElementById('occ-slider').value = currentMinOcc;
                        document.getElementById('slider-val').innerText = currentMinOcc;
                        nodesView.refresh();
                    }}
                    
                    // Focus and zoom on the selected node
                    network.selectNodes([node.id]);
                    network.focus(node.id, {{
                        scale: 1.2,
                        animation: {{
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }}
                    }});
                }});
                searchResults.appendChild(div);
            }});
        }});

        // Close search results when clicking outside
        document.addEventListener('click', function(e) {{
            var container = document.getElementById('search-container');
            if (container && !container.contains(e.target)) {{
                searchResults.style.display = 'none';
            }}
        }});

        // Sliders functionality
        var slider = document.getElementById('occ-slider');
        var sliderVal = document.getElementById('slider-val');
        var bridgeSlider = document.getElementById('bridge-slider');
        var bridgeSliderVal = document.getElementById('bridge-slider-val');
        var btnYes = document.getElementById('btn-show-isolated-yes');
        var btnNo = document.getElementById('btn-show-isolated-no');

        slider.addEventListener('input', function() {{
            currentMinOcc = parseInt(this.value);
            sliderVal.innerText = currentMinOcc;
            nodesView.refresh();
        }});

        bridgeSlider.addEventListener('input', function() {{
            currentMinAuthors = parseInt(this.value);
            bridgeSliderVal.innerText = currentMinAuthors;
            edgesView.refresh();
            nodesView.refresh();
        }});

        btnYes.addEventListener('click', function() {{
            if (!showIsolated) {{
                showIsolated = true;
                btnYes.classList.add('active');
                btnNo.classList.remove('active');
                nodesView.refresh();
            }}
        }});

        btnNo.addEventListener('click', function() {{
            if (showIsolated) {{
                showIsolated = false;
                btnYes.classList.remove('active');
                btnNo.classList.add('active');
                nodesView.refresh();
            }}
        }});
    </script>
</body>
</html>
"""
        with open(filepath, "w") as f:
            f.write(html_template)
        print(f"Interactive Cluster Author network saved to: {filepath}")

