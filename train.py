import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from efficientloader import EfficientChessDataset

# Define the model
class CompactChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(14, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 20480)  # Move scores

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(dataset, epochs=100):
    # Set up the device for GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # Initialize model, loss function, and optimizer
    model = CompactChessNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        # Generate a batch of training data
        board_states, move_labels = dataset.generate_training_batch()
        board_states, move_labels = board_states.to(device), move_labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(board_states)
        loss = criterion(outputs, move_labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # Prepare the model for quantization
    model.eval()  # Switch to evaluation mode
    model.cpu()  # Move to CPU for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # Set quantization config
    torch.quantization.prepare(model, inplace=True)  # Prepare model for quantization
    torch.quantization.convert(model, inplace=True)  # Convert to quantized model

    # Save the quantized model
    scripted_model = torch.jit.script(model)  # Script the model
    torch.jit.save(scripted_model, 'model_quantized.pt')  # Save the scripted model

    # # Quantize and save the model
    # quantized_model = torch.quantization.quantize_dynamic(
    #     model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    # )
    # torch.jit.save(torch.jit.script(quantized_model), 'model_quantized.pt')
    print("Model trained and saved.")

# Usage
dataset = EfficientChessDataset('data.pgn')
train_model(dataset)