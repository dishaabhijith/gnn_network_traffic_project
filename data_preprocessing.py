import pandas as pd
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def load_and_preprocess(csv_path, seq_len=5, window_size_seconds=5):
    """
    Preprocess your dataset with columns:
    - 'Time': relative time in seconds (float)
    - 'Source': source IP/device
    - 'Destination': destination IP/device
    - 'Length': packet length in bytes

    Aggregates bytes sent per node in time windows.

    Returns:
    - x_seq: tensor (1, seq_len, num_nodes, 1)
    - edge_index: tensor (2, num_edges)
    - targets: tensor (1, num_nodes, 1)
    """

    df = pd.read_csv(csv_path)

    # Column names based on your CSV
    src_col = 'Source'
    dst_col = 'Destination'
    time_col = 'Time'
    bytes_col = 'Length'

    # Convert 'Time' (seconds) to pandas timedelta for windowing
    df['timestamp'] = pd.to_timedelta(df[time_col], unit='s')

    # Get unique nodes
    nodes = pd.unique(df[[src_col, dst_col]].values.ravel())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)

    # Build undirected edges
    edges = set()
    for _, row in df.iterrows():
        src = node_to_idx[row[src_col]]
        dst = node_to_idx[row[dst_col]]
        edges.add((src, dst))
        edges.add((dst, src))
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    # Define time windows based on window_size_seconds
    start_time = df['timestamp'].min()
    time_windows = [start_time + pd.Timedelta(seconds=window_size_seconds*i) for i in range(seq_len+1)]

    # Initialize features: (seq_len, num_nodes, 1)
    features = np.zeros((seq_len, num_nodes, 1), dtype=np.float32)

    # Aggregate bytes sent per node per window
    for i in range(seq_len):
        window_df = df[(df['timestamp'] >= time_windows[i]) & (df['timestamp'] < time_windows[i+1])]
        for node in range(num_nodes):
            sent_bytes = window_df[window_df[src_col] == nodes[node]][bytes_col].sum()
            features[i, node, 0] = sent_bytes

    # Normalize features globally
    features = (features - features.min()) / (features.max() - features.min() + 1e-6)

    x_seq = torch.tensor(features).unsqueeze(0)  # add batch dim
    targets = x_seq[:, -1, :, :]  # predict last window bytes sent

    return x_seq, edge_index, targets

def add_knn_edges(x_seq, edge_index, k=5):
    """
    Adds KNN edges based on node feature similarity (Euclidean distance).
    Combines original edges with KNN edges.

    Args:
        x_seq: tensor (1, seq_len, num_nodes, num_features)
        edge_index: tensor (2, num_edges)
        k: number of nearest neighbors to connect

    Returns:
        combined_edge_index: tensor (2, num_edges + num_knn_edges)
    """
    features = x_seq[0, -1].numpy()  # Use last time window features, shape (num_nodes, num_features)

    knn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')  # +1 to include self
    knn.fit(features)

    distances, indices = knn.kneighbors(features)

    knn_edges = []
    num_nodes = features.shape[0]
    for i in range(num_nodes):
        for j in indices[i][1:]:  # skip self-loop
            knn_edges.append((i, j))
            knn_edges.append((j, i))  # add both directions for undirected graph

    knn_edge_index = torch.tensor(knn_edges, dtype=torch.long).t().contiguous()

    combined_edge_index = torch.cat([edge_index, knn_edge_index], dim=1)
    return combined_edge_index

def load_and_preprocess_with_knn(csv_path, seq_len=5, window_size_seconds=5, k=5):
    """
    Wrapper function to load data, preprocess, and add KNN edges.
    """
    x_seq, edge_index, targets = load_and_preprocess(csv_path, seq_len, window_size_seconds)
    combined_edge_index = add_knn_edges(x_seq, edge_index, k)
    return x_seq, combined_edge_index, targets


if __name__ == "__main__":
    x_seq, edge_index, targets = load_and_preprocess_with_knn('data/Midterm_53_group.csv')
    print("Input sequence shape:", x_seq.shape)
    print("Edge index shape:", edge_index.shape)
    print("Targets shape:", targets.shape)

