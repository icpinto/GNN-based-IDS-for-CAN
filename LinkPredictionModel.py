import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from torch import Tensor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch_geometric.transforms import RandomLinkSplit

def load_data(file_path, start_index, end_index):
    """Load data from CSV file and preprocess."""
    column_names = ['Timestamp', 'ID', 'DLC', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7', 'Flag']
    data_df = pd.read_csv(file_path, delimiter=r",", header=None, names=column_names)
    data = data_df[start_index:end_index]
    return data

def preprocess_data(data):
    """Preprocess data: Convert hex to decimal, drop NaN, and convert 'Flag' to binary."""
    hex_columns = ['ID', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7']
    for col in hex_columns:
        data[col] = data[col].apply(lambda x: int(x, 16) if isinstance(x, str) and all(c in '0123456789abcdefABCDEF' for c in x) else x)
    data.dropna(inplace=True)
    data['Flag'] = (data['Flag'] == 'T').astype(int)
    return data

def create_hetero_data(data):
    """Create HeteroData object with node features, indices, and edges."""
    can_traffic_node_features = torch.tensor(data.iloc[:, :-1].values.tolist(), dtype=torch.float)
    num_can_traffic_nodes = can_traffic_node_features.size(0)
    can_traffic_node_indices = torch.arange(num_can_traffic_nodes)
    flag_node_features = torch.tensor([[1], [0]], dtype=torch.float)
    num_flag_nodes = flag_node_features.size(0)
    flag_node_indices = torch.arange(num_flag_nodes)
    edges_can_traffic_flag = [(i, data.iloc[i, -1]) for i in range(num_can_traffic_nodes)]
    edges_can_traffic_flag = torch.tensor(edges_can_traffic_flag, dtype=torch.long).t().contiguous()
    edges_consecutive_can_traffic = [(i - 1, i) for i in range(1, num_can_traffic_nodes)]
    edges_consecutive_can_traffic = torch.tensor(edges_consecutive_can_traffic, dtype=torch.long).t().contiguous()
    
    hetero_data = HeteroData()
    hetero_data['CAN_traffic'].node_id = can_traffic_node_indices
    hetero_data['flag'].node_id = flag_node_indices
    hetero_data['CAN_traffic'].x = can_traffic_node_features
    hetero_data['flag'].x = flag_node_features
    hetero_data["CAN_traffic", "node", "flag"].edge_index = edges_can_traffic_flag
    hetero_data["CAN_traffic", "connect", "CAN_traffic"].edge_index = edges_consecutive_can_traffic
    
    return hetero_data, num_can_traffic_nodes, num_flag_nodes

def split_data(hetero_data):
    """Split data into training, validation, and testing sets."""
    transform = RandomLinkSplit(edge_types=("CAN_traffic", "node", "flag"))
    train_data, val_data, test_data = transform(hetero_data)
    return train_data, val_data, test_data

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_prob):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)  
        return x

class LinkPredictionClassifier(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, x: dict, edge_label_index: torch.Tensor) -> torch.Tensor:
        source_emb = x['CAN_traffic'][edge_label_index[0]]
        dest_emb = x['flag'][edge_label_index[1]]
        combined_emb = torch.cat([source_emb, dest_emb], dim=-1)
        out = F.relu(self.lin1(combined_emb))
        out = self.lin2(out)
        return out.view(-1)

class HeterogeneousLinkPredictionModel(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_prob,  num_CAN_traffic_nodes, num_flag_nodes, hetero_graph_metadata):
        super().__init__()
        self.CAN_traffic_lin = torch.nn.Linear(11, hidden_channels) 
        self.CAN_traffic_emb = nn.Embedding(num_CAN_traffic_nodes, hidden_channels)
        self.flag_emb = nn.Embedding(num_flag_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels,dropout_prob)
        self.gnn = to_hetero(self.gnn, hetero_graph_metadata)
        self.classifier = LinkPredictionClassifier(hidden_channels)

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {
            'CAN_traffic': self.CAN_traffic_emb(data['CAN_traffic'].node_id), 
            'flag': self.flag_emb(data['flag'].node_id)
        }
        updated_x = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(updated_x, data["CAN_traffic", "node", "flag"].edge_label_index)
        return pred

def train_model(model, train_data, lr=0.1, num_epochs=40):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        total_loss = total_examples = 0
        optimizer.zero_grad()
        pred = model(train_data)
        ground_truth = train_data["CAN_traffic", "node", "flag"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

def evaluate_model(model, data):
    preds = []
    ground_truths = []
    for data_split in data:
        with torch.no_grad():
            preds.append(model(data_split))
        ground_truths.append(data_split["CAN_traffic", "node", "flag"].edge_label.detach())
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    pred_binary = (pred >= 0.5).astype(int)
    auc = roc_auc_score(ground_truth, pred_binary)
    accuracy = accuracy_score(ground_truth, pred_binary)
    precision = precision_score(ground_truth, pred_binary)
    recall = recall_score(ground_truth, pred_binary)
    f1 = f1_score(ground_truth, pred_binary)
    confusion = confusion_matrix(ground_truth, pred_binary)
    print("Evaluation Metrics:")
    print("AUC:", auc)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(confusion)
    return auc, accuracy, precision, recall, f1, confusion
   
# Load data
data = load_data('/content/drive/My Drive/CANData/RPM_dataset.csv', 200000, 948900)

# Preprocess data
data = preprocess_data(data)

# Create HeteroData object
hetero_data, num_can_traffic_nodes, num_flag_nodes = create_hetero_data(data)

# Split data
train_data, val_data, test_data = split_data(hetero_data)

# Instantiate the model with the specified number of hidden channels and other necessary parameters
model = HeterogeneousLinkPredictionModel(hidden_channels=10,
                                         dropout_prob=0.01,
                                         num_CAN_traffic_nodes=num_can_traffic_nodes,
                                         num_flag_nodes=num_flag_nodes,
                                         hetero_graph_metadata=hetero_data.metadata())

# Training the model
train_model(model, train_data)

# Evaluating the model
data = [val_data]
auc, accuracy, precision, recall, f1, cm = evaluate_model(model, data)

