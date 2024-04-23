import pandas as pd
import torch
import networkx as nx
#from torch_geometric.data import Data
from torch.nn import Linear
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from itertools import combinations
import torch
import random
import numpy as np
from imblearn.over_sampling import SMOTE  # Import SMOTE for oversampling



def load_data(file_path):
    """Load data from CSV file."""
    column_names = ['Timestamp', 'ID', 'DLC', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7', 'Flag']
    data_df = pd.read_csv(file_path, delimiter=r",", header=None, names=column_names)
    return data_df

def preprocess_data(data):
    """Preprocess data."""
    hex_columns = ['ID', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7']
    for col in hex_columns:
        data[col] = data[col].apply(lambda x: int(x, 16) if isinstance(x, str) and all(c in '0123456789abcdefABCDEF' for c in x) else x)
    data.dropna(inplace=True)
    data['Flag'] = (data['Flag'] == 'T').astype(int)
    data['Timestamp_Diff'] = data['Timestamp'].diff().fillna(0)

    return data

def split_data(data, split_ratio=0.7):
    """Split data into training and testing sets."""
    train_data = data[:int(split_ratio * len(data))]
    test_data = data[int(split_ratio * len(data)):]
    return train_data, test_data

def create_mapping(data):
    """Create a mapping from unique ID indices to range [0, num_ID_nodes)."""
    id_mapping = {id: idx for idx, id in enumerate(data['ID'].unique())}
    return id_mapping

def replace_ids_with_mapped_values(data, id_mapping):
    """Replace ID indices with mapped values."""
    data['ID'] = data['ID'].map(id_mapping)
    return data

def create_graph(data):
    """Create a directed multigraph."""
    G = nx.from_pandas_edgelist(data, source='ID', target='ID', edge_attr=['Timestamp_Diff', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7', 'Flag'], create_using=nx.MultiDiGraph())
    return G

def convert_to_pyg_data(G, node_features):
    """Convert networkx multigraph to PyG Data object."""
    edge_index = []
    edge_attr = []
    y = []

    for u, v, data in G.edges(data=True):
        edge_index.append([u, v])
        if isinstance(data, dict):  # Check if data is a dictionary
            edge_attr.append([data.get('Timestamp_Diff', 0.0),
                              data.get('DATA0', 0.0), data.get('DATA1', 0.0),
                              data.get('DATA2', 0.0), data.get('DATA3', 0.0),
                              data.get('DATA4', 0.0), data.get('DATA5', 0.0),
                              data.get('DATA6', 0.0), data.get('DATA7', 0.0)])
            y.append(data.get('Flag', 0.0))  # Use default value 0.0 if 'Flag' key doesn't exist
        else:  # Scalar value case
            edge_attr.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Placeholder for edge attributes
            y.append(data['Flag'])  # Assuming scalar value is the label

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

     # Calculate node degrees
    node_degrees = [G.degree[node] for node in G.nodes()]
    node_features = torch.tensor(node_degrees, dtype=torch.float).view(-1, 1)  # Reshape to (num_nodes, 1)

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data



import torch.nn.utils as utils
def train_model(model, data, optimizer):
    """Train the model."""
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, data.edge_attr)
    target = data.y.view(-1, 1)
    loss = F.binary_cross_entropy(output, target)
    loss.backward()

    # Gradient clipping
    utils.clip_grad_norm_(model.parameters(), clip_value)

    optimizer.step()


    # Print train loss
    print(f"Train Loss: {loss.item():.4f}")

    return loss.item()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

def evaluate_model(model, data):
    """Evaluate the model."""
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.edge_attr)

    # Compute metrics
    predictions = (output > 0.5).float().cpu().numpy()
    ground_truth = data.y.view(-1, 1).cpu().numpy()
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    confusion = confusion_matrix(ground_truth, predictions)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth, output.cpu().numpy())

    # Compute AUC
    auc = roc_auc_score(ground_truth, output.cpu().numpy())

    # Print evaluation metrics
    print("Evaluation Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(confusion)
    print("AUC:", auc)

    return auc




class GCNWithEdgePrediction(nn.Module):
    def __init__(self, in_features, hidden_channels):
        super(GCNWithEdgePrediction, self).__init__()
        self.conv1 = GATConv(in_features, hidden_channels, edge_dim=9)
        self.lin = Linear(hidden_channels, 1)  # No need for concatenation

    def forward(self, x, edge_index, edge_attr):
        # Message passing on edges
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()

        # Message aggregation for connected nodes
        src, dest = edge_index
        # Compute the mean of node features for connected nodes
        # Change dim=1 to dim=0 if x is of shape (num_nodes, hidden_channels)
        out = (x[src] + x[dest]) / 2  # Mean aggregation

        out = self.lin(out)
        out = torch.sigmoid(out)
        return out


# Load and preprocess data
#data_df_DoS = load_data('/content/drive/My Drive/CANData/gear_dataset.csv')
#processed_DoS_data = preprocess_data(data_df_DoS)

data_df_gear = load_data('/content/drive/My Drive/CANData/RPM_dataset.csv')
proessed_gear_data = preprocess_data(data_df_gear[600000:948900])

#data_df_Fuzzy = load_data('/content/drive/My Drive/CANData/Fuzzy_dataset.csv')
#processed_Fuzzy_data = preprocess_data(data_df_Fuzzy[800000:948900])

# Split data into training and testing sets
train_data, test_data = split_data(proessed_gear_data)
#train_game_data, test_gear_data = split_data(processed_gear_data)
#train_Fuzzy_data, test_Fuzzy_data = split_data(processed_Fuzzy_data)

'''
# concatenating df1 and df2 along rows
train_data = pd.concat([train_DoS_data, train_game_data, train_Fuzzy_data], axis=0)
test_data = pd.concat([test_DoS_data, test_gear_data, test_Fuzzy_data], axis=0)'''

# Create mapping for ID indices
train_id_mapping = create_mapping(train_data)
test_id_mapping = create_mapping(test_data)

# Replace IDs with mapped values
train_data = replace_ids_with_mapped_values(train_data, train_id_mapping)
test_data = replace_ids_with_mapped_values(test_data, test_id_mapping)

# Create graphs for training and testing data
train_G = create_graph(train_data)
test_G = create_graph(test_data)

# Dummy node features for training and testing
train_x = torch.randn(len(train_G.nodes()), 16)
test_x = torch.randn(len(test_G.nodes()), 16)

# Convert graphs to PyG Data objects for training and testing
train_pyg_data = convert_to_pyg_data(train_G, train_x)
test_pyg_data = convert_to_pyg_data(test_G, test_x)

# Initialize model and optimizer
model = GCNWithEdgePrediction(1, hidden_channels=20)
optimizer = torch.optim.Adam(model.parameters(), lr=0.9)

clip_value = 1.0
# Train the model
for epoch in range(1, 20):
    train_loss = train_model(model, train_pyg_data, optimizer)

evaluate_model(model, test_pyg_data)

'''

from sklearn.model_selection import ParameterGrid

# Define a function for hyperparameter tuning
def hyperparameter_tuning(train_data, test_data, train_pyg_data, test_pyg_data):
    best_model = None
    best_performance = 0.0
    best_hyperparameters = {}

    # Define hyperparameter grid
    param_grid = {
        'lr': [0.001, 0.2, 0.01, 0.1,0.05],
        'hidden_channels': [16, 32, 64,10,40]
    }

    # Iterate over hyperparameter combinations
    for params in ParameterGrid(param_grid):
        print(params)
        # Initialize model and optimizer with current hyperparameters
        model = GCNWithEdgePrediction(1, hidden_channels=params['hidden_channels'])
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

        # Train the model
        for epoch in range(1, 20):
            train_loss = train_model(model, train_pyg_data, optimizer)

        # Evaluate the model
        performance = evaluate_model(model, test_pyg_data)

        # Update best model and hyperparameters if performance improves
        if performance > best_performance:
            best_performance = performance
            best_model = model
            best_hyperparameters = params

    # Print best hyperparameters and performance
    print("Best Hyperparameters:")
    print(best_hyperparameters)
    print("Best Performance:")
    print(best_performance)

    return best_model

# Perform hyperparameter tuning
best_model = hyperparameter_tuning(train_data, test_data, train_pyg_data, test_pyg_data)'''


