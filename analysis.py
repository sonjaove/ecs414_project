import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
import os
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_figure(filename):
    plt.gcf().savefig(
        os.path.join(RESULTS_DIR, filename),
        dpi=300,
        bbox_inches='tight'
    )

adjacency_matrix = pd.read_csv("unweighted_adjacency_matrix_nifty100.csv", index_col=0)
G = nx.from_pandas_adjacency(adjacency_matrix)

# Visualize and save the binary adjacency matrix.
plt.figure(figsize=(10, 8))
sns.heatmap(
    adjacency_matrix,
    cmap='Blues',
    vmin=0,
    vmax=1,
    xticklabels=False,
    yticklabels=False,
    cbar_kws={'label': 'Adjacency value'}
)
plt.title('Nifty 100 Stock Adjacency Matrix')
plt.xlabel('Stocks')
plt.ylabel('Stocks')
plt.tight_layout()
save_figure("adjacency_matrix_heatmap.png")

plt.figure(figsize=(16, 12))

pos = nx.spring_layout(G, seed=42, k=0.25)

# Download the sector information before it is used in the first network plot.
url = "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
response.raise_for_status()

nifty100 = pd.read_csv(StringIO(response.text))
nifty100['Ticker'] = nifty100['Symbol'] + '.NS'
sector_map = dict(zip(nifty100['Ticker'], nifty100['Industry']))

for stock in G.nodes():
    G.nodes[stock]['sector'] = sector_map.get(stock, 'Unknown')

sectors = nx.get_node_attributes(G, 'sector')
unique_sectors = sorted(set(sectors.values()))
palette = sns.color_palette('husl', len(unique_sectors))
sector_colors = dict(zip(unique_sectors, palette))

degrees = dict(G.degree())
node_sizes = [30 + degrees[n] * 5 for n in G.nodes()]

nx.draw_networkx_edges(
    G, pos,
    alpha=0.12,
    width=0.6,
    edge_color='black'
)

nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=[sector_colors.get(
        G.nodes[n].get('sector'), '#999999'
    ) for n in G.nodes()],
    alpha=0.85
)

# Label only the 10 most connected stocks
top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:10]

nx.draw_networkx_labels(
    G, pos,
    labels={n: n for n in top_nodes},
    font_size=8,
    font_weight='bold'
)

plt.title('Nifty 100 Stock Correlation Network', fontsize=17, pad=20)
plt.axis('off')
# Create industry legend
handles = [
    plt.Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor=color,
        markersize=9,
        label=sector
    )
    for sector, color in sector_colors.items()
]

plt.legend(
    handles=handles,
    title='Industry Sectors',
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    fontsize=9,
    title_fontsize=11,
    frameon=True
)

plt.tight_layout()
save_figure("nifty100_sector_network.png")
# plt.show()
# plt.tight_layout()
# plt.show()

''' comparision for thresholds, and based on this, 
we choose 0.2 as the threshold for the unweighted adjacency matrix for the Nifty 100 stock correlation network.'''
correlation_matrix = pd.read_csv("correlation_matrix.csv", index_col=0)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9]

results = []

for threshold in thresholds:
    adjacency = (correlation_matrix.abs() >= threshold).astype(int)
    adjacency = adjacency * (1 - np.eye(len(adjacency), dtype=int))

    G_temp = nx.from_pandas_adjacency(adjacency)

    results.append({
        'Threshold': threshold,
        'Nodes': G_temp.number_of_nodes(),
        'Edges': G_temp.number_of_edges(),
        'Density': nx.density(G_temp),
        'Avg Degree': np.mean([d for n, d in G_temp.degree()]),
        'Avg Clustering': nx.average_clustering(G_temp),
        'Components': nx.number_connected_components(G_temp)
    })

threshold_df = pd.DataFrame(results)
print(threshold_df.round(3))
threshold_df.to_csv(
    os.path.join(RESULTS_DIR, "threshold_comparison.csv"),
    index=False
)

sns.set_theme(style="whitegrid", context="notebook")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

metrics = ['Edges', 'Density', 'Avg Degree', 'Avg Clustering']

for ax, metric in zip(axes.flat, metrics):
    sns.lineplot(
        data=threshold_df,
        x='Threshold',
        y=metric,
        marker='o',
        linewidth=2.5,
        markersize=8,
        ax=ax
    )
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.set_xlabel('Correlation Threshold')
    ax.set_ylabel(metric)

fig.suptitle(
    'Network Properties Across Correlation Thresholds',
    fontsize=16,
    fontweight='bold'
)

plt.tight_layout()
save_figure("threshold_metrics.png")
# plt.show()

## A. Degree Distribution Plot & Interpretation

# Calculate degree of every stock
degrees = dict(G.degree())

degree_values = list(degrees.values())

# Display degree statistics
degree_df = pd.DataFrame({
    'Stock': list(degrees.keys()),
    'Degree': degree_values
})

print(degree_df.sort_values(by='Degree', ascending=False).head(10))
print(degree_df['Degree'].describe())
sns.set_theme(style="whitegrid", context="notebook")

degrees = dict(G.degree())

degree_df = pd.DataFrame({
    'Stock': list(degrees.keys()),
    'Degree': list(degrees.values())
})

print(degree_df.sort_values('Degree', ascending=False).head(10))
average_degree = degree_df['Degree'].mean()
median_degree = degree_df['Degree'].median()
degree_statistics = pd.DataFrame({
    'Metric': ['Average Degree', 'Median Degree', 'Minimum Degree', 'Maximum Degree'],
    'Value': [
        average_degree,
        median_degree,
        degree_df['Degree'].min(),
        degree_df['Degree'].max()
    ]
})
degree_statistics.to_csv(
    os.path.join(RESULTS_DIR, "degree_statistics.csv"),
    index=False
)
degree_df.sort_values(
    by='Degree',
    ascending=False
).to_csv(
    os.path.join(RESULTS_DIR, "degree_distribution.csv"),
    index=False
)
# gives the 15 stocks with the highest degree in the network
degree_df.sort_values(
    by='Degree',
    ascending=False
).head(15)
plt.figure(figsize=(10, 6))

sns.histplot(
    data=degree_df,
    x='Degree',
    bins=25,
    color='steelblue',
    edgecolor='black'
)

plt.axvline(
    average_degree,
    color='crimson',
    linestyle='--',
    linewidth=2,
    label=f'Average degree = {average_degree:.2f}'
)
plt.axvline(
    median_degree,
    color='darkgreen',
    linestyle=':',
    linewidth=2,
    label=f'Median degree = {median_degree:.0f}'
)
plt.title('Degree Distribution of Nifty 100 Stock Network')
plt.xlabel('Degree (Number of Connections)')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
save_figure("degree_histogram.png")
# plt.show()
# log-log plot for degree distribution
degree_counts = degree_df['Degree'].value_counts().sort_index()

# Remove degree zero and frequencies that cannot appear on a log scale
degree_counts = degree_counts[degree_counts.index > 0]

plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=degree_counts.index,
    y=degree_counts.values,
    color='darkred',
    s=70
)

if average_degree > 0:
    plt.axvline(
        average_degree,
        color='crimson',
        linestyle='--',
        linewidth=2,
        label=f'Average degree = {average_degree:.2f}'
    )

plt.xscale('log')
plt.yscale('log')

plt.title('Degree Distribution (Log-Log Scale)')
plt.xlabel('Degree (Log Scale)')
plt.ylabel('Frequency (Log Scale)')
plt.legend()

plt.tight_layout()
save_figure("degree_log_log_distribution.png")
# plt.show()

## B. Top-10 Centrality Leaderboard (The "Who's Who" of the Market)

# 1. Degree Centrality
degree_centrality = nx.degree_centrality(G)

# 2. Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(G)

# 3. Eigenvector Centrality
eigenvector_centrality = nx.eigenvector_centrality(
    G, max_iter=1000, tol=1e-06
)
# Create a centrality leaderboard
centrality_df = pd.DataFrame({
    'Stock': list(G.nodes()),
    'Degree Centrality': [degree_centrality[node] for node in G.nodes()],
    'Betweenness Centrality': [betweenness_centrality[node] for node in G.nodes()],
    'Eigenvector Centrality': [eigenvector_centrality[node] for node in G.nodes()]
})

centrality_df.to_csv(
    os.path.join(RESULTS_DIR, "centrality_leaderboard.csv"),
    index=False
)

print(centrality_df.head())
# Get the Top 10 stocks for each measure
# Top 10 by Degree Centrality
top_degree = centrality_df.sort_values(
    by='Degree Centrality', ascending=False
).head(10)

# Top 10 by Betweenness Centrality
top_betweenness = centrality_df.sort_values(
    by='Betweenness Centrality', ascending=False
).head(10)

# Top 10 by Eigenvector Centrality
top_eigenvector = centrality_df.sort_values(
    by='Eigenvector Centrality', ascending=False
).head(10)

print("TOP 10 - DEGREE CENTRALITY")
print(top_degree.to_string(index=False))

print("\nTOP 10 - BETWEENNESS CENTRALITY")
print(top_betweenness.to_string(index=False))

print("\nTOP 10 - EIGENVECTOR CENTRALITY")
print(top_eigenvector.to_string(index=False))
# Visualise the Top 10 leaderboard 
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

centralities = [
    (top_degree, 'Degree Centrality', 'Top 10 - Degree Centrality'),
    (top_betweenness, 'Betweenness Centrality', 'Top 10 - Betweenness Centrality'),
    (top_eigenvector, 'Eigenvector Centrality', 'Top 10 - Eigenvector Centrality')
]

for ax, (data, column, title) in zip(axes, centralities):
    sns.barplot(
        data=data,
        x=column,
        y='Stock',
        hue='Stock',
        palette='viridis',
        legend=False,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel('Centrality Score')
    ax.set_ylabel('')

plt.tight_layout()
save_figure("top10_centrality_leaderboard.png")
# plt.show()

## C. Local Clustering Coefficient & Sector Homophily

# Average clustering coefficient of the entire network
avg_clustering = nx.average_clustering(G)

print("Average Clustering Coefficient:", avg_clustering)
# Calculate individual stock clustering coefficients
clustering_coefficients = nx.clustering(G)

clustering_df = pd.DataFrame({
    'Stock': list(clustering_coefficients.keys()),
    'Clustering Coefficient': list(clustering_coefficients.values())
})

# Sort stocks by clustering coefficient
clustering_df = clustering_df.sort_values(
    by='Clustering Coefficient',
    ascending=False
)

clustering_df.to_csv(
    os.path.join(RESULTS_DIR, "clustering_coefficients.csv"),
    index=False
)

print(clustering_df.head(15))
# Visualise the top 15 stocks
plt.figure(figsize=(10, 6))

sns.barplot(
    data=clustering_df.head(15),
    x='Clustering Coefficient',
    y='Stock',
    hue='Stock',
    palette='viridis',
    legend=False
)

plt.title('Top 15 Stocks by Clustering Coefficient')
plt.axvline(
    avg_clustering,
    color='crimson',
    linestyle='--',
    linewidth=2,
    label=f'Network average = {avg_clustering:.3f}'
)
plt.legend()
plt.tight_layout()
save_figure("top15_clustering_coefficients.png")
# plt.show()

### Sector homophily - Are stocks more likely to be connected to other stocks belonging to the same industry?
# Sector attributes and colors were initialized before the first plot.
print(pd.Series(
    nx.get_node_attributes(G, 'sector')
).value_counts())
sector_counts = pd.Series(
    nx.get_node_attributes(G, 'sector')
).value_counts().rename_axis('Sector').reset_index(name='Count')
sector_counts.to_csv(
    os.path.join(RESULTS_DIR, "sector_counts.csv"),
    index=False
)
sector_assortativity = nx.attribute_assortativity_coefficient(
    G, 'sector'
)

print("Sector Homophily:", sector_assortativity)
with open(os.path.join(RESULTS_DIR, "network_summary.txt"), 'w') as summary_file:
    summary_file.write(
        f"Average Degree: {average_degree:.6f}\n"
        f"Median Degree: {median_degree:.6f}\n"
        f"Average Clustering Coefficient: {avg_clustering:.6f}\n"
        f"Sector Homophily: {sector_assortativity:.6f}\n"
    )
# Visualize the network with nodes colored by sector
plt.figure(figsize=(16, 12))

pos = nx.spring_layout(G, seed=42)

sectors = nx.get_node_attributes(G, 'sector')
unique_sectors = sorted(set(sectors.values()))

palette = sns.color_palette('husl', len(unique_sectors))
sector_colors = dict(zip(unique_sectors, palette))

node_colors = [sector_colors[sectors[node]] for node in G.nodes()]

nx.draw_networkx(
    G,
    pos,
    node_color=node_colors,
    node_size=100,
    with_labels=False,
    edge_color='gray',
    alpha=0.7,
    width=0.3
)
# Create legend handles
handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color, markersize=9, label=sector)
    for sector, color in sector_colors.items()
]

plt.legend(handles=handles, title="Industry Sector",
           bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title(
    f'Nifty 100 Stock Network by Industry Sector\n'
    f'Sector homophily: {sector_assortativity:.3f}'
)
plt.tight_layout()

save_figure("nifty100_network_by_industry_sector.png")
# plt.show()

