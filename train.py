import torch
import torch.nn as nn
import torch.optim as optim
from data_preprocessing import load_and_preprocess_with_knn
from temporal_gcn import TemporalGCN

def train():
    # Load and preprocess data with KNN edges added
    x_seq, edge_index, targets = load_and_preprocess_with_knn('data/Midterm_53_group.csv', seq_len=5, window_size_seconds=5, k=5)

    # Initialize model with 1 input feature per node
    model = TemporalGCN(in_channels=1, hidden_channels=16, lstm_hidden=32, out_channels=1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_seq, edge_index)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'model.pth')
    print("Training complete and model saved.")

if __name__ == "__main__":
    train()

