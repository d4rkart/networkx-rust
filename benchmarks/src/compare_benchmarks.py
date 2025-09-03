#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import re

def load_benchmark_results(python_file, rust_file, python_detailed=None, rust_detailed=None):
    """Load benchmark results from CSV files."""
    python_df = pd.read_csv(python_file)
    rust_df = pd.read_csv(rust_file)
    
    # Parse benchmark names to categorize them
    def parse_benchmark_name(name):
        if 'creation' in name:
            match = re.search(r'(.+)_(\d+)_(\d+)', name)
            if match:
                category = match.group(1)
                nodes = int(match.group(2))
                edges = int(match.group(3))
                return category, nodes, edges
        elif 'add_' in name:
            match = re.search(r'add_(\d+)_(\w+)', name)
            if match:
                count = int(match.group(1))
                item_type = match.group(2)
                return f"add_{item_type}", count, None
        else:
            return name, None, None
        
        return name, None, None
    
    # Add parsed information as columns
    for df in [python_df, rust_df]:
        df['Category'], df['Size1'], df['Size2'] = zip(*df['Benchmark'].apply(parse_benchmark_name))
    
    # Load detailed results if available
    detailed_data = {}
    if python_detailed and os.path.exists(python_detailed):
        py_detailed = pd.read_csv(python_detailed)
        detailed_data['python'] = py_detailed
    
    if rust_detailed and os.path.exists(rust_detailed):
        rs_detailed = pd.read_csv(rust_detailed)
        detailed_data['rust'] = rs_detailed
    
    return python_df, rust_df, detailed_data

def merge_results(python_df, rust_df):
    """Merge Python and Rust results for comparison."""
    # Create a merged dataframe
    merged = pd.merge(
        python_df, rust_df, 
        on=['Category', 'Size1', 'Size2'], 
        suffixes=('_py', '_rs')
    )
    
    # Calculate speedup
    merged['Speedup'] = merged['Time (ms)_py'] / merged['Time (ms)_rs']
    
    return merged

def plot_comparisons(merged_df, output_dir='.'):
    """Create comparison plots for visualization."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Enhanced style for plots
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (14, 8)
    
    # Filter for different types of benchmarks
    creation_df = merged_df[merged_df['Category'].str.contains('creation')]
    operation_df = merged_df[~merged_df['Category'].str.contains('creation') & 
                            ~merged_df['Category'].str.contains('add_')]
    addition_df = merged_df[merged_df['Category'].str.contains('add_')]
    
    # Sort for better visualization
    creation_df = creation_df.sort_values(by=['Size1', 'Size2'])
    
    # Color scheme
    python_color = '#3498db'  # Blue
    rust_color = '#e74c3c'    # Red
    
    # Set style for all plots
    plot_kwargs = {
        'edgecolor': 'black',
        'linewidth': 0.5,
        'alpha': 0.8,
    }
    
    # 1. Creation Time Comparison
    plt.figure(figsize=(16, 10))
    bar_width = 0.35
    index = np.arange(len(creation_df))
    
    # Create bars
    py_bars = plt.bar(index, creation_df['Time (ms)_py'], width=bar_width, label='Python (NetworkX)', color=python_color, **plot_kwargs)
    rs_bars = plt.bar(index + bar_width, creation_df['Time (ms)_rs'], width=bar_width, label='Rust (networkx-rs)', color=rust_color, **plot_kwargs)
    
    # Add value labels on top of bars
    for bar in py_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    for bar in rs_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    # Add detailed annotations
    max_py_time = creation_df['Time (ms)_py'].max()
    max_rs_time = creation_df['Time (ms)_rs'].max()
    max_time = max(max_py_time, max_rs_time)
    
    # Calculate average speedup for graph creation
    avg_speedup = creation_df['Time (ms)_py'].mean() / creation_df['Time (ms)_rs'].mean()
    plt.annotate(f'Average Speedup: {avg_speedup:.1f}x', 
                xy=(0.5, 0.97), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                ha='center', fontsize=12)
    
    plt.xlabel('Graph Size (Nodes, Edges)', fontweight='bold', fontsize=14)
    plt.ylabel('Time (ms)', fontweight='bold', fontsize=14)
    plt.title('Graph Creation Performance: NetworkX vs networkx-rs\nLower bars indicate better performance', 
              fontsize=18, fontweight='bold', pad=20)
    
    plt.xticks(index + bar_width/2, [f"{row['Category'].replace('_creation', '')} ({row['Size1']:,} nodes, {row['Size2']:,} edges)" 
                                    for _, row in creation_df.iterrows()], rotation=45, ha='right')
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # # Add explanatory text
    # plt.figtext(0.5, 0.01, 
    #            "This chart compares the time taken to create graphs of various sizes in both Python (NetworkX) and Rust (networkx-rs).\n"
    #            "Bars show average time in milliseconds (lower is better). Rust consistently outperforms Python across all graph sizes.",
    #            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.savefig(os.path.join(output_dir, 'creation_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 1.1 Graph creation speedup by size (log scale for better visibility)
    plt.figure(figsize=(14, 8))
    
    plt.scatter(creation_df['Size1'], creation_df['Speedup'], 
                s=creation_df['Size2']/50, alpha=0.7, 
                c=creation_df['Speedup'], cmap='viridis', 
                edgecolors='black', linewidths=1)
    
    for i, row in creation_df.iterrows():
        plt.annotate(f"{row['Category'].replace('_creation', '')}\n({row['Size1']:,} nodes, {row['Size2']:,} edges)",
                    (row['Size1'], row['Speedup']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal performance')
    plt.colorbar(label='Speedup factor (higher is better for Rust)')
    plt.xlabel('Number of Nodes (log scale)', fontweight='bold', fontsize=14)
    plt.ylabel('Speedup (Python time / Rust time)', fontweight='bold', fontsize=14)
    plt.title('Graph Creation Speedup vs Graph Size\nBubble size represents number of edges', 
              fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'creation_speedup_by_size.png'), dpi=300, bbox_inches='tight')
    
    # 2. Operation Time Comparison
    if not operation_df.empty:
        plt.figure(figsize=(16, 10))
        index = np.arange(len(operation_df))
        
        # Sort operations by python time for better visualization
        operation_df = operation_df.sort_values(by='Time (ms)_py')
        
        # Create bars
        py_bars = plt.bar(index, operation_df['Time (ms)_py'], width=bar_width, label='Python (NetworkX)', color=python_color, **plot_kwargs)
        rs_bars = plt.bar(index + bar_width, operation_df['Time (ms)_rs'], width=bar_width, label='Rust (networkx-rs)', color=rust_color, **plot_kwargs)
        
        # Add value labels on top of bars
        for bar in py_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9, rotation=45)
        
        for bar in rs_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9, rotation=45)
        
        # Calculate average speedup for operations
        avg_speedup = operation_df['Time (ms)_py'].mean() / operation_df['Time (ms)_rs'].mean()
        plt.annotate(f'Average Speedup: {avg_speedup:.1f}x', 
                    xy=(0.5, 0.97), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    ha='center', fontsize=12)
        
        plt.xlabel('Graph Operation', fontweight='bold', fontsize=14)
        plt.ylabel('Time (ms)', fontweight='bold', fontsize=14)
        plt.title('Graph Algorithm Performance: NetworkX vs networkx-rs\nLower bars indicate better performance', 
                  fontsize=18, fontweight='bold', pad=20)
        
        # More descriptive operation names
        operation_names = []
        for _, row in operation_df.iterrows():
            if row['Size1'] is not None:
                operation_names.append(f"{row['Category']} ({row['Size1']} nodes)")
            else:
                operation_names.append(row['Category'])
        
        plt.xticks(index + bar_width/2, operation_names, rotation=45, ha='right')
        plt.legend(loc='upper left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add explanatory text
        # plt.figtext(0.5, 0.01, 
        #            "This chart compares the execution time of various graph operations in both Python (NetworkX) and Rust (networkx-rs).\n"
        #            "Bars show average time in milliseconds (lower is better). Operations are sorted by Python execution time.",
        #            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.savefig(os.path.join(output_dir, 'operation_comparison.png'), dpi=300, bbox_inches='tight')
        
        # 2.1 Log scale for operations with large differences
        plt.figure(figsize=(16, 10))
        
        py_bars = plt.bar(index, operation_df['Time (ms)_py'], width=bar_width, label='Python (NetworkX)', color=python_color, **plot_kwargs)
        rs_bars = plt.bar(index + bar_width, operation_df['Time (ms)_rs'], width=bar_width, label='Rust (networkx-rs)', color=rust_color, **plot_kwargs)
        
        # Add value labels on top of bars
        for bar in py_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9, rotation=45)
        
        for bar in rs_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9, rotation=45)
        
        plt.xlabel('Graph Operation', fontweight='bold', fontsize=14)
        plt.ylabel('Time (ms) - Log Scale', fontweight='bold', fontsize=14)
        plt.title('Graph Algorithm Performance: NetworkX vs networkx-rs (Log Scale)\nHighlights performance differences across operations of varying costs', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.yscale('log')
        plt.xticks(index + bar_width/2, operation_names, rotation=45, ha='right')
        plt.legend(loc='upper left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add explanatory text
        # plt.figtext(0.5, 0.01, 
                #    "This chart uses a logarithmic scale to better visualize performance differences across operations with varying execution times.\n"
                #    "The log scale helps highlight relative performance differences even when absolute times vary by orders of magnitude.",
                #    ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.savefig(os.path.join(output_dir, 'operation_comparison_log.png'), dpi=300, bbox_inches='tight')
        
        # 2.2 Operation speedup chart (speedup for each operation)
        plt.figure(figsize=(14, 8))
        
        # Sort by speedup for this chart
        op_speedup_df = operation_df.sort_values(by='Speedup', ascending=False)
        speedup_bars = plt.bar(range(len(op_speedup_df)), op_speedup_df['Speedup'], 
                               color=[rust_color if x > 1 else python_color for x in op_speedup_df['Speedup']])
        
        # Add value labels
        for bar in speedup_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}x', ha='center', va='bottom')
        
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Equal performance')
        
        plt.xlabel('Graph Operation', fontweight='bold', fontsize=14)
        plt.ylabel('Speedup Factor (Python time / Rust time)', fontweight='bold', fontsize=14)
        plt.title('Operation-specific Performance Speedup\nHigher bars indicate better Rust performance', 
                 fontsize=18, fontweight='bold', pad=20)
        
        operation_speedup_names = []
        for _, row in op_speedup_df.iterrows():
            if row['Size1'] is not None:
                operation_speedup_names.append(f"{row['Category']} ({row['Size1']} nodes)")
            else:
                operation_speedup_names.append(row['Category'])
        
        plt.xticks(range(len(op_speedup_df)), operation_speedup_names, rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Add explanatory text
        # plt.figtext(0.5, 0.01, 
        #            "This chart shows the speedup factor for each operation (Python time divided by Rust time).\n"
        #            "Higher bars (> 1.0) indicate Rust is faster; lower bars (< 1.0) indicate Python is faster for that operation.",
        #            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.savefig(os.path.join(output_dir, 'operation_speedup.png'), dpi=300, bbox_inches='tight')
    
    # 3. Addition Time Comparison
    if not addition_df.empty:
        plt.figure(figsize=(14, 8))
        index = np.arange(len(addition_df))
        
        # Sort addition operations
        addition_df = addition_df.sort_values(by=['Size1', 'Category'])
        
        # Create bars
        py_bars = plt.bar(index, addition_df['Time (ms)_py'], width=bar_width, label='Python (NetworkX)', color=python_color, **plot_kwargs)
        rs_bars = plt.bar(index + bar_width, addition_df['Time (ms)_rs'], width=bar_width, label='Rust (networkx-rs)', color=rust_color, **plot_kwargs)
        
        # Add value labels on top of bars
        for bar in py_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9, rotation=45)
        
        for bar in rs_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9, rotation=45)
        
        # Calculate average speedup for additions
        avg_speedup = addition_df['Time (ms)_py'].mean() / addition_df['Time (ms)_rs'].mean()
        plt.annotate(f'Average Speedup: {avg_speedup:.1f}x', 
                    xy=(0.5, 0.97), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    ha='center', fontsize=12)
        
        plt.xlabel('Graph Modification Operation', fontweight='bold', fontsize=14)
        plt.ylabel('Time (ms)', fontweight='bold', fontsize=14)
        plt.title('Graph Modification Performance: Adding Nodes/Edges\nGrouped by graph size and operation', 
                 fontsize=18, fontweight='bold', pad=20)
        
        # More descriptive operation names for x-axis
        addition_labels = []
        for _, row in addition_df.iterrows():
            # Include the base graph size in the label
            size_info = row['Size1']
            if 'add_' in row['Category']:
                parts = row['Category'].split('_')
                if len(parts) >= 2 and parts[1].isdigit():
                    num_items = int(parts[1])
                    if 'nodes' in row['Category']:
                        item_type = 'nodes'
                    else:
                        item_type = 'edges'
                    addition_labels.append(f"Add {num_items} {item_type}\nto graph with {size_info:,} nodes")
                else:
                    addition_labels.append(f"{row['Category']}\n({size_info:,} nodes)")
            else:
                addition_labels.append(f"{row['Category']}\n({size_info:,} nodes)")
        
        plt.xticks(index + bar_width/2, addition_labels, rotation=45, ha='right')
        plt.legend(loc='upper left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Add explanatory text
        # plt.figtext(0.5, 0.01, 
        #            "This chart compares the time taken to add nodes or edges to existing graphs of various sizes.\n"
        #            "The performance difference highlights memory allocation and data structure efficiency between implementations.",
        #            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.savefig(os.path.join(output_dir, 'addition_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 4. Overall Speedup
    plt.figure(figsize=(16, 12))  # Larger figure for more bars
    
    # Sort by speedup
    sorted_df = merged_df.sort_values(by='Speedup', ascending=False)
    
    # Color bars based on speedup
    colors = [rust_color if x > 1 else python_color for x in sorted_df['Speedup']]
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(sorted_df)), sorted_df['Speedup'], color=colors, 
                   edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width > 1:
            label_text = f'{width:.1f}x faster in Rust'
            label_color = 'black'
        else:
            label_text = f'{1/width:.1f}x faster in Python'
            label_color = 'black'
        
        plt.text(max(width + 0.1, 1.1), i, label_text, 
                va='center', color=label_color, fontsize=8)
    
    plt.xscale('log')
    plt.xlabel('Speedup Factor (Python time / Rust time)', fontweight='bold', fontsize=14)
    plt.title('Performance Comparison: Rust vs Python\nAll benchmarked operations sorted by speedup factor', 
             fontsize=18, fontweight='bold', pad=20)
    
    # Create better benchmark labels
    benchmark_labels = []
    for _, row in sorted_df.iterrows():
        benchmark = row['Benchmark_py']
        if row['Size1'] is not None and row['Size2'] is not None:
            # This is a graph creation or operation on a specific graph
            benchmark_labels.append(f"{benchmark} ({row['Size1']:,} nodes, {row['Size2']:,} edges)")
        elif row['Size1'] is not None:
            # This is likely an operation on a graph with only Size1 specified
            benchmark_labels.append(f"{benchmark} ({row['Size1']:,} nodes)")
        else:
            benchmark_labels.append(benchmark)
    
    plt.yticks(range(len(sorted_df)), benchmark_labels)
    plt.axvline(x=1, color='black', linestyle='--', label='Equal performance')
    plt.grid(True, axis='x', which='both', linestyle='--', alpha=0.7)
    
    # Add median and geometric mean speedup
    geomean_speedup = np.exp(np.mean(np.log(sorted_df['Speedup'])))
    median_speedup = sorted_df['Speedup'].median()
    
    annotation_text = (f"Median Speedup: {median_speedup:.1f}x\n"
                       f"Geometric Mean Speedup: {geomean_speedup:.1f}x\n"
                       f"Operations where Rust is faster: {sum(sorted_df['Speedup'] > 1)}/{len(sorted_df)}")
    
    plt.annotate(annotation_text, 
                xy=(0.01, 0.01), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                ha='left', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    # Add explanatory text
    # plt.figtext(0.5, 0.01, 
    #            "This chart shows the speedup factor for all benchmarked operations (Python time divided by Rust time).\n"
    #            "Bars to the right of the center line (> 1.0) indicate Rust is faster; bars to the left (< 1.0) indicate Python is faster.",
    #            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.savefig(os.path.join(output_dir, 'speedup.png'), dpi=300, bbox_inches='tight')
    
    # 5. Graph size vs performance comparison
    if len(creation_df) > 1:
        plt.figure(figsize=(14, 10))
        
        # Create scatter plot with more information
        graph_sizes = creation_df['Size1'] * creation_df['Size2']
        speedups = creation_df['Speedup']
        
        # Create size-based bubbles
        sc = plt.scatter(
            graph_sizes,  # Graph complexity (nodes * edges)
            speedups,
            c=speedups,
            cmap='viridis',
            s=np.sqrt(graph_sizes) * 3,  # Scale bubble size with sqrt of complexity
            alpha=0.7,
            edgecolors='black'
        )
        
        # Add data point labels
        for i, row in creation_df.iterrows():
            complexity = row['Size1'] * row['Size2']
            speedup = row['Speedup']
            plt.annotate(f"({row['Size1']:,} nodes, {row['Size2']:,} edges)",
                        (complexity, speedup),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, color='black', 
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.7))
        
        # Add trendline
        if len(graph_sizes) > 1:
            z = np.polyfit(np.log10(graph_sizes), np.log10(speedups), 1)
            p = np.poly1d(z)
            x_trend = np.logspace(np.log10(min(graph_sizes)), np.log10(max(graph_sizes)), 100)
            plt.plot(x_trend, 10**p(np.log10(x_trend)), "r--", alpha=0.7, 
                    label=f"Trend: speedup ~ size^{z[0]:.2f}")
            
            # Add trend equation
            plt.annotate(f"Trend: Speedup ∝ (Graph Size)^{z[0]:.2f}", 
                         xy=(0.05, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                         ha='left', va='top', fontsize=12)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Graph Complexity (Nodes × Edges)', fontweight='bold', fontsize=14)
        plt.ylabel('Speedup Factor (Python time / Rust time)', fontweight='bold', fontsize=14)
        plt.title('Rust Performance Advantage vs Graph Size\nBubble size indicates relative graph complexity', 
                 fontsize=18, fontweight='bold', pad=20)
        
        plt.colorbar(sc, label='Speedup Factor')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Equal performance')
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        # Add explanatory text
        # plt.figtext(0.5, 0.01, 
        #            "This chart shows how Rust's performance advantage scales with graph size and complexity.\n"
        #            "The trend line indicates the relationship between graph size and speedup, highlighting how performance differences change with scale.",
        #            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.savefig(os.path.join(output_dir, 'size_vs_speedup.png'), dpi=300, bbox_inches='tight')
    
    print(f"Charts saved to {output_dir}")
    return sorted_df

def create_html_report(merged_df, output_file='benchmark_report.html', detailed_data=None):
    """Create an HTML report of benchmark results."""
    # Format the merged dataframe for display
    display_df = merged_df.copy()
    display_df['Python Time (ms)'] = display_df['Time (ms)_py'].round(2)
    display_df['Rust Time (ms)'] = display_df['Time (ms)_rs'].round(2)
    display_df['Speedup'] = display_df['Speedup'].round(2)
    
    # Add "Test Description" column with more detailed explanations
    display_df['Test Description'] = ''
    for i, row in display_df.iterrows():
        benchmark = row['Benchmark_py']
        if 'creation' in row['Category']:
            if 'graph' in row['Category']:
                display_df.at[i, 'Test Description'] = f"Creating an undirected graph with {row['Size1']:,} nodes and {row['Size2']:,} edges"
            else:
                display_df.at[i, 'Test Description'] = f"Creating a directed graph with {row['Size1']:,} nodes and {row['Size2']:,} edges"
        elif 'shortest_path' in row['Category']:
            if 'length' in row['Category']:
                display_df.at[i, 'Test Description'] = f"Finding shortest path length between two nodes in a graph with {row['Size1']:,} nodes"
            elif 'all_pairs' in row['Category']:
                display_df.at[i, 'Test Description'] = f"Computing all-pairs shortest paths in a graph with {row['Size1']:,} nodes"
            else:
                display_df.at[i, 'Test Description'] = f"Finding shortest path between two nodes in a graph with {row['Size1']:,} nodes"
        elif 'dfs' in row['Category']:
            display_df.at[i, 'Test Description'] = f"Performing depth-first search on a graph with {row['Size1']:,} nodes"
        elif 'bfs' in row['Category']:
            display_df.at[i, 'Test Description'] = f"Performing breadth-first search on a graph with {row['Size1']:,} nodes"
        elif 'pagerank' in row['Category']:
            display_df.at[i, 'Test Description'] = f"Computing PageRank on a graph with {row['Size1']:,} nodes"
        elif 'components' in row['Category']:
            if 'strongly' in row['Category']:
                display_df.at[i, 'Test Description'] = f"Finding strongly connected components in a directed graph with {row['Size1']:,} nodes"
            else:
                display_df.at[i, 'Test Description'] = f"Finding connected components in an undirected graph with {row['Size1']:,} nodes"
        elif 'add_' in row['Category']:
            parts = row['Category'].split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                num_items = int(parts[1])
                if 'nodes' in row['Category']:
                    display_df.at[i, 'Test Description'] = f"Adding {num_items} nodes to a graph with {row['Size1']:,} nodes"
                elif 'edges' in row['Category']:
                    display_df.at[i, 'Test Description'] = f"Adding {num_items} edges to a graph with {row['Size1']:,} nodes"
                else:
                    display_df.at[i, 'Test Description'] = f"Adding {num_items} items to a graph with {row['Size1']:,} nodes"
            else:
                display_df.at[i, 'Test Description'] = f"Graph modification operation on a graph with {row['Size1']:,} nodes"
    
    # Calculate summary statistics
    avg_speedup = merged_df['Speedup'].mean()
    geomean_speedup = np.exp(np.mean(np.log(merged_df['Speedup'])))
    median_speedup = merged_df['Speedup'].median()
    max_speedup = merged_df['Speedup'].max()
    min_speedup = merged_df['Speedup'].min()
    rust_faster_count = sum(merged_df['Speedup'] > 1)
    python_faster_count = sum(merged_df['Speedup'] < 1)
    equal_count = sum(merged_df['Speedup'] == 1)
    
    # Group by operation type for summarized view
    operation_types = {
        'Graph Creation': merged_df[merged_df['Category'].str.contains('creation')],
        'Path Finding': merged_df[merged_df['Category'].str.contains('shortest_path')],
        'Graph Traversal': merged_df[(merged_df['Category'] == 'dfs') | (merged_df['Category'] == 'bfs')],
        'Advanced Algorithms': merged_df[(merged_df['Category'] == 'pagerank') | (merged_df['Category'].str.contains('components'))],
        'Graph Modification': merged_df[merged_df['Category'].str.contains('add_')]
    }
    
    operation_summaries = {}
    for op_type, op_df in operation_types.items():
        if not op_df.empty:
            op_avg_speedup = op_df['Speedup'].mean()
            op_geomean_speedup = np.exp(np.mean(np.log(op_df['Speedup'])))
            op_speedup_range = (op_df['Speedup'].min(), op_df['Speedup'].max())
            operation_summaries[op_type] = {
                'avg_speedup': op_avg_speedup,
                'geomean_speedup': op_geomean_speedup,
                'min_speedup': op_speedup_range[0],
                'max_speedup': op_speedup_range[1],
                'count': len(op_df)
            }
    
    # Create sorted dataframes for display
    by_operation_df = display_df.sort_values(by=['Category', 'Size1', 'Size2'])
    by_speedup_df = display_df.sort_values(by='Speedup', ascending=False)
    
    # Select display columns and rename as needed
    by_operation_df = by_operation_df.rename(columns={'Benchmark_py': 'Benchmark'})
    by_speedup_df = by_speedup_df.rename(columns={'Benchmark_py': 'Benchmark'})
    
    display_cols = ['Benchmark', 'Test Description', 'Python Time (ms)', 'Rust Time (ms)', 'Speedup']
    by_operation_df = by_operation_df[display_cols]
    by_speedup_df = by_speedup_df[display_cols]
    
    # Create HTML with embedded images and interactive tables
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NetworkX vs networkx-rs Benchmark Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; line-height: 1.6; }}
            h1 {{ color: #2c3e50; text-align: center; margin-top: 30px; }}
            h2 {{ color: #3498db; margin-top: 25px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h3 {{ color: #2c3e50; margin-top: 20px; }}
            p {{ margin: 12px 0; color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; position: sticky; top: 0; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #e5e5e5; }}
            .speedup {{ text-align: center; }}
            .faster {{ color: #e74c3c; font-weight: bold; }}
            .slower {{ color: #3498db; font-weight: bold; }}
            .equal {{ color: #2c3e50; }}
            .image-container {{ margin: 30px 0; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; display: block; margin: 10px auto; }}
            .summary {{ background-color: #e8f4f8; padding: 15px; border-left: 5px solid #3498db; margin: 20px 0; border-radius: 0 5px 5px 0; }}
            .stat-card {{ background-color: white; border-radius: 8px; padding: 15px; margin: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: inline-block; width: 280px; text-align: center; }}
            .stat-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; color: #3498db; }}
            .stat-label {{ font-size: 14px; color: #7f8c8d; }}
            .stats-container {{ display: flex; flex-wrap: wrap; justify-content: center; margin: 20px 0; }}
            .chart-description {{ color: #555; font-style: italic; text-align: center; margin: 5px 0 20px 0; max-width: 800px; margin-left: auto; margin-right: auto; }}
            .caption {{ font-size: 14px; color: #666; text-align: center; margin-top: 10px; font-style: italic; }}
            .conclusion {{ margin: 30px 0; padding: 20px; background-color: #f8f9fa; border-radius: 5px; border-left: 5px solid #2c3e50; }}
            footer {{ margin-top: 30px; text-align: center; color: #7f8c8d; padding: 10px; border-top: 1px solid #ddd; }}
            .tooltip {{ position: relative; display: inline-block; cursor: help; border-bottom: 1px dotted #3498db; }}
            .tooltip .tooltiptext {{ visibility: hidden; width: 200px; background-color: #555; color: #fff; text-align: center; border-radius: 6px; padding: 5px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -100px; opacity: 0; transition: opacity 0.3s; }}
            .tooltip:hover .tooltiptext {{ visibility: visible; opacity: 1; }}
            .tabs {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; margin-top: 20px; border-radius: 5px 5px 0 0; }}
            .tab {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 17px; }}
            .tab:hover {{ background-color: #ddd; }}
            .tab.active {{ background-color: #3498db; color: white; }}
            .tabcontent {{ display: none; padding: 20px; border: 1px solid #ccc; border-top: none; border-radius: 0 0 5px 5px; }}
            .visible {{ display: block; }}
        </style>
        <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tablinks = document.getElementsByClassName("tab");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}
        
        // Show the first tab by default
        window.onload = function() {{
            document.getElementsByClassName("tab")[0].click();
        }};
        </script>
    </head>
    <body>
        <h1>NetworkX vs networkx-rs Benchmark Comparison</h1>
        
        <div class="summary">
            <p>This report compares the performance of the Python NetworkX library with the Rust networkx-rs implementation
            across various graph operations and graph sizes. The benchmarks measure execution time for key graph algorithms
            and operations to quantify performance differences between the implementations.</p>
        </div>
        
        <h2>Performance Overview</h2>
        
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-label">Geometric Mean Speedup</div>
                <div class="stat-value">{geomean:.2f}x</div>
                <div class="stat-label">Rust over Python</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">Maximum Speedup</div>
                <div class="stat-value">{max_speedup:.2f}x</div>
                <div class="stat-label">Best case for Rust</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">Operations Faster in Rust</div>
                <div class="stat-value">{rust_faster} / {total_ops}</div>
                <div class="stat-label">Operations</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">Median Speedup</div>
                <div class="stat-value">{median:.2f}x</div>
                <div class="stat-label">Middle of distribution</div>
            </div>
        </div>
        
        <h2>Performance by Operation Type</h2>
        <table>
            <tr>
                <th>Operation Type</th>
                <th>Operations</th>
                <th>Average Speedup</th>
                <th>Geometric Mean</th>
                <th>Range</th>
            </tr>
    """
    
    # Format the template with the statistics
    html = html_template.format(
        geomean=geomean_speedup,
        max_speedup=max_speedup,
        rust_faster=rust_faster_count,
        total_ops=len(merged_df),
        median=median_speedup
    )
    
    # Add operation type summary rows
    op_type_html = ""
    for op_type, stats in operation_summaries.items():
        op_type_html += f"""
            <tr>
                <td>{op_type}</td>
                <td>{stats['count']}</td>
                <td>{stats['avg_speedup']:.2f}x</td>
                <td>{stats['geomean_speedup']:.2f}x</td>
                <td>{stats['min_speedup']:.2f}x - {stats['max_speedup']:.2f}x</td>
            </tr>
        """
    
    html += op_type_html
    
    # Add rest of the HTML
    html += """
        </table>
        
        <h2>Detailed Performance Comparison</h2>
        
        <div class="tabs">
            <button class="tab" onclick="openTab(event, 'ByOperation')">By Operation Type</button>
            <button class="tab" onclick="openTab(event, 'BySpeedup')">Sorted by Speedup</button>
        </div>
        
        <div id="ByOperation" class="tabcontent">
            <h3>Results Grouped by Operation Type</h3>
            <table>
                <tr>
                    <th>Benchmark</th>
                    <th>Description</th>
                    <th>Python Time (ms)</th>
                    <th>Rust Time (ms)</th>
                    <th>Speedup</th>
                </tr>
    """
    
    # Add rows for by-operation table
    for _, row in by_operation_df.iterrows():
        speedup_class = "faster" if row['Speedup'] > 1 else "slower" if row['Speedup'] < 1 else "equal"
        html += f"""
                <tr>
                    <td>{row['Benchmark']}</td>
                    <td>{row['Test Description']}</td>
                    <td>{row['Python Time (ms)']:.2f}</td>
                    <td>{row['Rust Time (ms)']:.2f}</td>
                    <td class="speedup {speedup_class}">{row['Speedup']:.2f}x</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div id="BySpeedup" class="tabcontent">
            <h3>Results Ranked by Performance Difference</h3>
            <table>
                <tr>
                    <th>Benchmark</th>
                    <th>Description</th>
                    <th>Python Time (ms)</th>
                    <th>Rust Time (ms)</th>
                    <th>Speedup</th>
                </tr>
    """
    
    # Add rows for by-speedup table
    for _, row in by_speedup_df.iterrows():
        speedup_class = "faster" if row['Speedup'] > 1 else "slower" if row['Speedup'] < 1 else "equal"
        html += f"""
                <tr>
                    <td>{row['Benchmark']}</td>
                    <td>{row['Test Description']}</td>
                    <td>{row['Python Time (ms)']:.2f}</td>
                    <td>{row['Rust Time (ms)']:.2f}</td>
                    <td class="speedup {speedup_class}">{row['Speedup']:.2f}x</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <h2>Visualization and Analysis</h2>
    """
    
    # Add images and descriptions
    image_sections = [
        {
            'title': 'Graph Creation Performance',
            'file': 'creation_comparison.png',
            'description': 'This chart compares the time taken to create graphs of various sizes in both Python and Rust. Lower bars indicate better performance.'
        },
        {
            'title': 'Creation Speedup by Graph Size',
            'file': 'creation_speedup_by_size.png',
            'description': 'This scatter plot shows how the speedup factor changes with graph size. The bubble size represents the number of edges in each graph.'
        },
        {
            'title': 'Graph Operation Performance',
            'file': 'operation_comparison.png',
            'description': 'This chart compares the execution time of various graph operations. Lower bars indicate better performance.'
        },
        {
            'title': 'Graph Operation Performance (Log Scale)',
            'file': 'operation_comparison_log.png',
            'description': 'This chart uses a logarithmic scale to better visualize performance differences across operations with varying execution times.'
        },
        {
            'title': 'Operation-specific Speedup',
            'file': 'operation_speedup.png',
            'description': 'This chart shows the speedup factor for each operation. Higher bars indicate Rust is faster; lower bars indicate Python is faster.'
        },
        {
            'title': 'Graph Modification Performance',
            'file': 'addition_comparison.png',
            'description': 'This chart compares the time taken to add nodes or edges to existing graphs. The performance difference highlights memory allocation and data structure efficiency.'
        },
        {
            'title': 'Overall Performance Comparison',
            'file': 'speedup.png',
            'description': 'This chart shows the speedup factor for all benchmarked operations. Bars to the right of the center line indicate Rust is faster; bars to the left indicate Python is faster.'
        },
        {
            'title': 'Performance Scaling with Graph Size',
            'file': 'size_vs_speedup.png',
            'description': 'This chart shows how Rust\'s performance advantage scales with graph size and complexity. The trend line indicates how the speedup changes with increasing graph size.'
        }
    ]
    
    for section in image_sections:
        html += f"""
        <div class="image-container">
            <h3>{section['title']}</h3>
            <p class="chart-description">{section['description']}</p>
            <img src="{section['file']}" alt="{section['title']}">
        </div>
        """
    
    # Add HTML section for run consistency if detailed data is available
    if detailed_data and 'python' in detailed_data and 'rust' in detailed_data:
        html += """
        <h2>Benchmark Run Consistency Analysis</h2>
        <p class="summary">
            This section analyzes the consistency of benchmark runs, showing how stable the performance is across multiple executions.
            Lower standard deviation and coefficient of variation indicate more consistent and reliable benchmark results.
        </p>
        
        <div class="image-container">
            <h3>Standard Deviation Comparison</h3>
            <p class="chart-description">
                Standard deviation measures the absolute variation between runs in milliseconds. Lower values indicate more consistent benchmark results.
            </p>
            <img src="standard_deviation_comparison.png" alt="Standard Deviation Comparison">
        </div>
        
        <div class="image-container">
            <h3>Coefficient of Variation</h3>
            <p class="chart-description">
                Coefficient of variation (CV) is the standard deviation divided by the mean, expressed as a percentage. 
                It provides a relative measure of run-to-run variation, allowing for comparison between benchmarks of different scales.
            </p>
            <img src="coefficient_of_variation.png" alt="Coefficient of Variation">
        </div>
        
        <div class="image-container">
            <h3>Min-Max Range Analysis</h3>
            <p class="chart-description">
                This chart shows the range between minimum and maximum run times as a percentage of the mean.
                A smaller range indicates more consistent performance across all runs.
            </p>
            <img src="min_max_range.png" alt="Min-Max Range Analysis">
        </div>
        
        <div class="image-container">
            <h3>Consistency Comparison Scatter Plot</h3>
            <p class="chart-description">
                This scatter plot compares Python and Rust implementation consistency. Points below the diagonal line indicate Rust is more consistent,
                while points above the line indicate Python is more consistent for that particular benchmark.
            </p>
            <img src="consistency_comparison.png" alt="Consistency Comparison">
        </div>
        """
    
    # Add conclusion
    html += f"""
        <div class="conclusion">
            <h2>Summary and Conclusions</h2>
            <p>Based on the benchmarks performed, the Rust implementation (networkx-rs) shows an average speedup of {geomean_speedup:.2f}x 
            over the Python implementation (NetworkX) across all tested operations. Out of {len(merged_df)} operations benchmarked, 
            Rust was faster in {rust_faster_count} cases.</p>
            
            <p>Key observations:</p>
            <ul>
                <li>The performance advantage of Rust tends to increase with graph size, with a maximum speedup of {max_speedup:.2f}x.</li>
                <li>Graph creation operations show consistent speedup across all graph sizes.</li>
                <li>The most significant performance differences are observed in operations that are computationally intensive or involve heavy memory operations.</li>
                <li>For very small graphs, the performance difference is less pronounced, but still favors Rust in most cases.</li>
            </ul>
        </div>
        
        <footer>
            <p>Benchmark report generated for networkx-rs vs NetworkX performance comparison.</p>
            <p>Rust implementation: networkx-rs | Python implementation: NetworkX</p>
        </footer>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"HTML report saved to {output_file}")
    return True

def plot_run_consistency(detailed_data, output_dir='.'):
    """Create plots visualizing run consistency and variation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for plots
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Collect data for plotting
    if 'python' not in detailed_data or 'rust' not in detailed_data:
        print("Missing detailed data for one or both implementations. Skipping consistency plots.")
        return
    
    py_data = detailed_data['python']
    rs_data = detailed_data['rust']
    
    # Match benchmarks that exist in both datasets
    common_benchmarks = set(py_data['Benchmark']).intersection(set(rs_data['Benchmark']))
    
    if not common_benchmarks:
        print("No common benchmarks found in detailed data. Skipping consistency plots.")
        return
    
    # Prepare data frames for matched benchmarks
    py_filtered = py_data[py_data['Benchmark'].isin(common_benchmarks)]
    rs_filtered = rs_data[rs_data['Benchmark'].isin(common_benchmarks)]
    
    # 1. Standard Deviation Comparison
    plt.figure(figsize=(16, 12))
    
    # Merge on benchmark name
    sd_comparison = pd.merge(
        py_filtered[['Benchmark', 'StdDev (ms)']],
        rs_filtered[['Benchmark', 'StdDev (ms)']],
        on='Benchmark',
        suffixes=('_py', '_rs')
    )
    
    # Sort for better visualization
    sd_comparison = sd_comparison.sort_values(by='StdDev (ms)_py', ascending=False)
    
    # Create bars
    bar_width = 0.4
    index = np.arange(len(sd_comparison))
    
    python_color = '#3498db'  # Blue
    rust_color = '#e74c3c'    # Red
    
    py_bars = plt.bar(index, sd_comparison['StdDev (ms)_py'], width=bar_width, label='Python (NetworkX)', 
                    color=python_color, edgecolor='black', linewidth=0.5, alpha=0.8)
    rs_bars = plt.bar(index + bar_width, sd_comparison['StdDev (ms)_rs'], width=bar_width, label='Rust (networkx-rs)', 
                     color=rust_color, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Benchmark Operation', fontweight='bold', fontsize=14)
    plt.ylabel('Standard Deviation (ms)', fontweight='bold', fontsize=14)
    plt.title('Run-to-Run Variation: Standard Deviation Comparison\nLower values indicate more consistent benchmark results',
             fontsize=18, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for bar in py_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    for bar in rs_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    # Add benchmark labels
    plt.xticks(index + bar_width/2, sd_comparison['Benchmark'], rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the chart
    plt.savefig(os.path.join(output_dir, 'standard_deviation_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 2. Coefficient of Variation Comparison (StdDev / Mean)
    plt.figure(figsize=(16, 12))
    
    # Calculate coefficient of variation
    cv_comparison = pd.merge(
        py_filtered[['Benchmark', 'Mean (ms)', 'StdDev (ms)']],
        rs_filtered[['Benchmark', 'Mean (ms)', 'StdDev (ms)']],
        on='Benchmark',
        suffixes=('_py', '_rs')
    )
    
    cv_comparison['CV_py'] = cv_comparison['StdDev (ms)_py'] / cv_comparison['Mean (ms)_py'] * 100  # as percentage
    cv_comparison['CV_rs'] = cv_comparison['StdDev (ms)_rs'] / cv_comparison['Mean (ms)_rs'] * 100  # as percentage
    
    # Sort by Python CV
    cv_comparison = cv_comparison.sort_values(by='CV_py', ascending=False)
    
    # Create bars
    py_bars = plt.bar(index, cv_comparison['CV_py'], width=bar_width, label='Python (NetworkX)', 
                    color=python_color, edgecolor='black', linewidth=0.5, alpha=0.8)
    rs_bars = plt.bar(index + bar_width, cv_comparison['CV_rs'], width=bar_width, label='Rust (networkx-rs)', 
                     color=rust_color, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Benchmark Operation', fontweight='bold', fontsize=14)
    plt.ylabel('Coefficient of Variation (%)', fontweight='bold', fontsize=14)
    plt.title('Run-to-Run Variation: Coefficient of Variation Comparison\nLower values indicate more consistent benchmark results relative to mean time',
             fontsize=18, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for bar in py_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9, rotation=45)
    
    for bar in rs_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9, rotation=45)
    
    # Add benchmark labels
    plt.xticks(index + bar_width/2, cv_comparison['Benchmark'], rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the chart
    plt.savefig(os.path.join(output_dir, 'coefficient_of_variation.png'), dpi=300, bbox_inches='tight')
    
    # 3. Min-Max Range Comparison
    plt.figure(figsize=(16, 12))
    
    # Calculate the range as percentage of mean
    range_comparison = pd.merge(
        py_filtered[['Benchmark', 'Mean (ms)', 'Min (ms)', 'Max (ms)']],
        rs_filtered[['Benchmark', 'Mean (ms)', 'Min (ms)', 'Max (ms)']],
        on='Benchmark',
        suffixes=('_py', '_rs')
    )
    
    range_comparison['Range_py'] = range_comparison['Max (ms)_py'] - range_comparison['Min (ms)_py']
    range_comparison['Range_rs'] = range_comparison['Max (ms)_rs'] - range_comparison['Min (ms)_rs']
    
    range_comparison['Range%_py'] = range_comparison['Range_py'] / range_comparison['Mean (ms)_py'] * 100
    range_comparison['Range%_rs'] = range_comparison['Range_rs'] / range_comparison['Mean (ms)_rs'] * 100
    
    # Sort by Python range
    range_comparison = range_comparison.sort_values(by='Range%_py', ascending=False)
    
    # Create bars
    py_bars = plt.bar(index, range_comparison['Range%_py'], width=bar_width, label='Python (NetworkX)', 
                     color=python_color, edgecolor='black', linewidth=0.5, alpha=0.8)
    rs_bars = plt.bar(index + bar_width, range_comparison['Range%_rs'], width=bar_width, label='Rust (networkx-rs)', 
                      color=rust_color, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Benchmark Operation', fontweight='bold', fontsize=14)
    plt.ylabel('Min-Max Range (% of mean)', fontweight='bold', fontsize=14)
    plt.title('Run-to-Run Variation: Min-Max Range Comparison\nLower values indicate more consistent benchmark results',
              fontsize=18, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for bar in py_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9, rotation=45)
    
    for bar in rs_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9, rotation=45)
    
    # Add benchmark labels
    plt.xticks(index + bar_width/2, range_comparison['Benchmark'], rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the chart
    plt.savefig(os.path.join(output_dir, 'min_max_range.png'), dpi=300, bbox_inches='tight')
    
    # 4. Consistency Rank Comparison
    # Calculate a consistency rank based on CV - lower is better
    plt.figure(figsize=(16, 12))
    
    # Create a scatter plot of CV values
    plt.scatter(cv_comparison['CV_py'], cv_comparison['CV_rs'], 
               s=100, alpha=0.7, c=range(len(cv_comparison)), cmap='viridis', edgecolors='black')
    
    # Add diagonal line for equal performance
    max_cv = max(cv_comparison['CV_py'].max(), cv_comparison['CV_rs'].max())
    plt.plot([0, max_cv*1.1], [0, max_cv*1.1], 'k--', alpha=0.7)
    
    # Add labels for points
    for i, row in cv_comparison.iterrows():
        plt.annotate(row['Benchmark'], 
                    (row['CV_py'], row['CV_rs']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
    
    # Add region labels
    plt.annotate('Rust more consistent', xy=(max_cv*0.75, max_cv*0.25), fontsize=12, 
                ha='center', va='center', bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))
    plt.annotate('Python more consistent', xy=(max_cv*0.25, max_cv*0.75), fontsize=12, 
                ha='center', va='center', bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))
    
    plt.xlabel('Python Coefficient of Variation (%)', fontweight='bold', fontsize=14)
    plt.ylabel('Rust Coefficient of Variation (%)', fontweight='bold', fontsize=14)
    plt.title('Run-to-Run Consistency Comparison\nPoints below the diagonal line indicate Rust has more consistent results',
             fontsize=18, fontweight='bold', pad=20)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Calculate percentage of benchmarks where Rust is more consistent
    rust_more_consistent = sum(cv_comparison['CV_rs'] < cv_comparison['CV_py']) / len(cv_comparison) * 100
    plt.annotate(f'Rust more consistent in {rust_more_consistent:.1f}% of benchmarks', 
                xy=(0.02, 0.98), xycoords='axes fraction', fontsize=12, 
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8),
                ha='left', va='top')
    
    # Save the chart
    plt.savefig(os.path.join(output_dir, 'consistency_comparison.png'), dpi=300, bbox_inches='tight')
    
    print(f"Consistency charts saved to {output_dir}")
    
    return {
        'cv_comparison': cv_comparison,
        'sd_comparison': sd_comparison,
        'range_comparison': range_comparison
    }

def main():
    parser = argparse.ArgumentParser(description='Compare NetworkX and networkx-rs benchmark results')
    parser.add_argument('--python', type=str, default='results/python_benchmark_results.csv',
                        help='Python benchmark results CSV file')
    parser.add_argument('--rust', type=str, default='results/rust_benchmark_results.csv',
                        help='Rust benchmark results CSV file')
    parser.add_argument('--python-detailed', type=str, default=None,
                        help='Python detailed benchmark results CSV file')
    parser.add_argument('--rust-detailed', type=str, default=None,
                        help='Rust detailed benchmark results CSV file')
    parser.add_argument('--output-dir', type=str, default='charts',
                        help='Directory to save output charts')
    parser.add_argument('--report', type=str, default='benchmark_report.html',
                        help='Output HTML report file')
    
    args = parser.parse_args()
    
    # Set default detailed files if not specified
    if args.python_detailed is None:
        args.python_detailed = args.python.replace('.csv', '_detailed.csv')
    
    if args.rust_detailed is None:
        args.rust_detailed = args.rust.replace('.csv', '_detailed.csv')
    
    # Load benchmark results
    python_df, rust_df, detailed_data = load_benchmark_results(
        args.python, args.rust, args.python_detailed, args.rust_detailed)
    
    # Merge results
    merged_df = merge_results(python_df, rust_df)
    
    # Create visualization
    sorted_df = plot_comparisons(merged_df, args.output_dir)
    
    # Plot run consistency if detailed data is available
    if detailed_data:
        plot_run_consistency(detailed_data, args.output_dir)
    
    # Create HTML report
    create_html_report(sorted_df, args.report, detailed_data)

if __name__ == "__main__":
    main() 