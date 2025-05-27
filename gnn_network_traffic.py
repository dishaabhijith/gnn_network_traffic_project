import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_channels, output_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def main():
    num_nodes = 4
    num_features = 3

    np.random.seed(42)
    features = np.random.rand(num_nodes, num_features).astype(np.float32)

    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                               [1, 0, 2, 1, 3, 2]], dtype=torch.long)

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    x = torch.tensor(features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    model = GCN(num_node_features=num_features, hidden_channels=16, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    target = torch.tensor([[0.5], [0.6], [0.7], [0.8]], dtype=torch.float)

    model.train()
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(data)
        mse = mean_squared_error(target.numpy(), pred.numpy())
        print(f"\nMean Squared Error on synthetic data: {mse:.4f}")

        print("\nPredicted traffic values per node:")
        for i, p in enumerate(pred.numpy()):
            print(f"Node {i}: {p[0]:.4f}")

if __name__ == "__main__":
    main()
