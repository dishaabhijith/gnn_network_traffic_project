from flask import Flask, render_template, request, jsonify
import torch
from temporal_gcn import TemporalGCN
from data_preprocessing import load_and_preprocess_with_knn
import pandas as pd

app = Flask(__name__)

# Load and preprocess data once at startup with KNN edges added
csv_path = 'data/Midterm_53_group.csv'
x_seq, edge_index, _ = load_and_preprocess_with_knn(csv_path, seq_len=5, window_size_seconds=5, k=5)

# Extract unique nodes (IPs) from dataset for labeling predictions
def get_nodes_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    src_col = 'Source'
    dst_col = 'Destination'
    nodes_set = set(pd.unique(df[[src_col, dst_col]].values.ravel()))
    return list(nodes_set)

nodes = get_nodes_from_csv(csv_path)

# Initialize model with input feature size = 1 (adjust if you have more features)
model = TemporalGCN(in_channels=1, hidden_channels=16, lstm_hidden=32, out_channels=1)

# Load trained model weights
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    with torch.no_grad():
        pred = model(x_seq, edge_index)  # Shape: (1, num_nodes, 1)
    pred_list = pred.squeeze().tolist()  # Convert to flat list

    # Pair node names with predictions, round for readability
    results = [{"node": node, "predicted_traffic": round(value, 4)} for node, value in zip(nodes, pred_list)]

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

