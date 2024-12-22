import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
from torch.cuda.amp import GradScaler, autocast

# ----------------------------------
# define device
# ----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------
# Q1 Functions
# ----------------------------------

# 1.1 Show VGG16 Model Structure
def load_and_show_vgg16():
    """
    Corresponding to Button 1.1: Show Structure
    """
    model = torchvision.models.vgg16_bn(weights=None).to(device)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1).to(device)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10).to(device)
    print("VGG16 Model Structure:")
    summary(model, input_size=(1, 32, 32))

# Dataset Preparation for Training
def prepare_mnist_dataloader(batch_size=64):
    """
    Prepare DataLoader for MNIST Dataset
    """
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader

# 1.2 Train and Save Logs
def train_vgg16_and_save_logs(epochs=30, log_path="mnist_vgg16_log.pth", model_path="best_mnist_vgg16.pth"):
    """
    Train VGG16 model and save logs and best model weights
    """
    train_loader, val_loader = prepare_mnist_dataloader()
    model = torchvision.models.vgg16_bn(weights=None).to(device)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1).to(device)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scaler = GradScaler()  # For mixed precision training
    logs = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0

    torch.backends.cudnn.benchmark = True  # Enable cuDNN optimization

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():  # Mixed precision context
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        logs["train_loss"].append(train_loss / len(train_loader))
        logs["val_loss"].append(val_loss / len(val_loader))
        logs["train_acc"].append(train_acc)
        logs["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

    torch.save(logs, log_path)
    print(f"Training complete. Logs saved to {log_path}, best model saved to {model_path}")

# 1.2 Show Training Logs
def show_accuracy_and_loss(log_path="mnist_vgg16_log.pth"):
    """
    Corresponding to Button 1.2: Show Acc and Loss
    """
    if not os.path.exists(log_path):
        print("Log file not found!")
        return

    logs = torch.load(log_path)
    epochs = range(len(logs["train_loss"]))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, logs["train_loss"], label="Train Loss")
    plt.plot(epochs, logs["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, logs["train_acc"], label="Train Accuracy")
    plt.plot(epochs, logs["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

# 1.3 Predict with VGG16
def predict_with_vgg16(model_path, image_path):
    """
    Corresponding to Button 1.3: Predict
    """
    if not os.path.exists(model_path):
        print("Model file not found! Please train the model first.")
        return None

    model = torchvision.models.vgg16_bn(weights=None).to(device)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1).to(device)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities).item()

    print(f"Predicted number: {predicted_class}")
    return predicted_class

# ----------------------------------
# Q2 Placeholder Functions
# ----------------------------------
def prepare_cat_dog_dataloader(dataset_path, batch_size=32):
    pass

def show_resnet50_architecture():
    pass

def train_or_load_resnet50(model_path, dataset_path, epochs=10, batch_size=32):
    pass

def predict_with_resnet50(model_path, image_path):
    pass

# ----------------------------------
# Main Execution for Training
# ----------------------------------
if __name__ == "__main__":
    if not os.path.exists("mnist_vgg16_log.pth") or not os.path.exists("best_mnist_vgg16.pth"):
        print("Training VGG16 for Q1...")
        train_vgg16_and_save_logs()
    else:
        print("Logs and model already exist. No training required.")
