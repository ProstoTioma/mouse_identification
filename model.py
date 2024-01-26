import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# Generate some dummy data for demonstration
# Replace this with your actual data loading and preprocessing logic
def generate_data():
    data_path = "mouse_movement_filtered.txt"
    data_path_fake = "mouse_movement_filtered_fake.txt"
    real_data = []
    fake_data = []
    labels = []

    with open(data_path, 'r') as f:
        lines = [line.strip().split(",") for line in f.readlines()]
    for line in lines:
        line = [int(value) for value in line]
        real_data.append(line)
        labels.append(1)

    with open(data_path_fake, 'r') as f:
        lines = [line.strip().split(",") for line in f.readlines()]

    for line in lines:
        line = [int(value) for value in line]
        fake_data.append(line)
        labels.append(0)

    real_data = np.array(real_data)
    fake_data = np.array(fake_data)
    total_data = [real_data, fake_data]
    labels = np.array(labels)
    # labels.resize((labels.size, 1))
    labels = np.array([1, 0])

    return total_data, labels


class MouseRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MouseRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, hn = self.rnn(x)
        output = self.fc(hn[-1])
        return self.sigmoid(output)


# Hyperparameters
input_size = 2  # (x, y) coordinates
hidden_size = 64
output_size = 1  # Binary classification

# Initialize the model
model = MouseRNN(input_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate data
data, labels = generate_data()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {accuracy * 100:.2f}%')
