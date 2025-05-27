import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class TemporalGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, lstm_hidden, out_channels):
        super(TemporalGCN, self).__init__()
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.lstm = nn.LSTM(hidden_channels, lstm_hidden, batch_first=True)
        self.linear = nn.Linear(lstm_hidden, out_channels)

    def forward(self, x_seq, edge_index):
        batch_size, seq_len, num_nodes, feat_dim = x_seq.size()
        gcn_out_seq = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :, :].reshape(-1, feat_dim)
            gcn_out = self.gcn(x_t, edge_index)
            gcn_out_seq.append(gcn_out.view(batch_size, num_nodes, -1))
        gcn_out_seq = torch.stack(gcn_out_seq, dim=1)
        lstm_in = gcn_out_seq.view(batch_size * num_nodes, seq_len, -1)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = lstm_out[:, -1, :]
        out = self.linear(lstm_out)
        out = out.view(batch_size, num_nodes, -1)
        return out
