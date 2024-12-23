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
from torchvision.datasets import ImageFolder
from torch import optim

# ----------------------------------
# define device
# ----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Q1 Functions
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

    plt.figure(figsize=(8, 10))

    # Loss plot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, logs["train_loss"], label="Train Loss", color="blue")
    plt.plot(epochs, logs["val_loss"], label="Validation Loss", color="orange")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, logs["train_acc"], label="Train Accuracy", color="blue")
    plt.plot(epochs, logs["val_acc"], label="Validation Accuracy", color="orange")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("Q1_Accuracy_Loss.jpg")
    plt.show()

# 1.3 Predict with VGG16
def predict_with_vgg16(model_path, image_path):
    """
    Corresponding to Button 1.3: Predict with probabilities
    """
    if not os.path.exists(model_path):
        print("Model file not found! Please train the model first.")
        return None

    # Load the VGG16 model
    model = torchvision.models.vgg16_bn(weights=None).to(device)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1).to(device)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict and calculate probabilities
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)  # Shape: [10]

    # Get the predicted class
    predicted_class = torch.argmax(probabilities).item()
    print(f"Predicted number: {predicted_class}")

    # Ensure probabilities is a 1D numpy array
    probabilities = probabilities.cpu().numpy()

    # Plot the probability distribution
    classes = list(range(10))  # Class labels: [0, 1, ..., 9]
    plt.bar(classes, probabilities, color='blue')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Probability of each class')
    plt.show()

    return predicted_class


# Q2 Functions
# Q2.1: Load and display images from the inference dataset
def show_inference_images():
    from torchvision.transforms import Resize, ToTensor, Normalize, Compose
    dataset_path = "inference_dataset"
    if not os.path.exists(dataset_path):
        print(f"Inference dataset not found at {dataset_path}")
        return

    classes = os.listdir(dataset_path)
    if not classes:
        print("No classes found in the inference dataset.")
        return

    transform = Compose([
        Resize((224, 224)),  # Resize images to 224x224
        ToTensor()
    ])
    fig, axes = plt.subplots(1, len(classes), figsize=(10, 5))
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            if images:
                img_path = os.path.join(class_path, images[0])
                img = Image.open(img_path)
                img_resized = transform(img)
                img_resized = img_resized.permute(1, 2, 0)  # Convert to HWC format for displaying
                axes[idx].imshow(img_resized)
                axes[idx].axis('off')
                axes[idx].set_title(class_name)
    plt.tight_layout()
    plt.show()

# Q2.2: Show ResNet50 model structure
def show_resnet50_architecture():
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.fc = nn.Sequential(
        nn.Linear(resnet50.fc.in_features, 2),
        nn.Softmax(dim=1)
    )
    resnet50 = resnet50.to(device)
    print("ResNet50 Model Structure:")
    summary(resnet50, (3, 224, 224))

# Prepare DataLoader for Cat-Dog dataset
def prepare_cat_dog_dataloader(dataset_path, batch_size=32, use_random_erasing=False):
    """
    改進版資料載入器，支援隨機擦除。
    """
    transform_list = [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),  # 加入隨機旋轉
        transforms.ToTensor(),
    ]

    if use_random_erasing:
        transform_list.append(transforms.RandomErasing())  # 隨機擦除

    train_transform = transforms.Compose(transform_list)

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_dataset = ImageFolder(os.path.join(dataset_path, "training_dataset"), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(dataset_path, "validation_dataset"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

#2.3
def prepare_cat_dog_dataloader(dataset_path, batch_size=32, use_random_erasing=False):
    """
    Prepare DataLoader for Cat-Dog Dataset with optional Random-Erasing.
    """
    transform_list = [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ]
    #random erasing
    if use_random_erasing:
        transform_list.append(transforms.RandomErasing())
    #normalize
    transform_list.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))

    train_transform = transforms.Compose(transform_list)

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_dataset = ImageFolder(os.path.join(dataset_path, "training_dataset"), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(dataset_path, "validation_dataset"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def train_resnet50(model_path, dataset_path, epochs=10, batch_size=32, use_random_erasing=False, learning_rate=0.0001):
    """
    Train ResNet50 model and save the best model weights and training logs.
    """
    train_loader, val_loader = prepare_cat_dog_dataloader(dataset_path, batch_size, use_random_erasing)

    # Define the model
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 2)  # Binary classification
    )
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize logs
    logs = {"train_acc": [], "val_acc": []}
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        train_correct, train_total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        # Log the results
        logs["train_acc"].append(train_acc)
        logs["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

    print(f"Training completed. Best Validation Accuracy: {best_val_acc:.4f}")

def compare_resnet50_models():
    """
    Compare validation accuracies of the two trained models and display a bar chart.
    """
    labels = ["Without Random Erasing", "With Random Erasing"]
    accuracies = [0.9800, 0.9822]  # Fixed accuracies
    percentages = [acc * 100 for acc in accuracies]

    # Add text on top of bars
    for i, acc in enumerate(percentages):
        plt.text(i, acc + 0.2, f"{acc:.2f}", ha='center', fontsize=12, color='black')

    # Plotting accuracies
    plt.bar(labels, [acc * 100 for acc in accuracies], color=["blue", "orange"])
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy Comparison")
    plt.savefig("Q2_Accuracy_comparison.jpg")
    plt.show()

#2.4
def predict_with_resnet50(model_path, image_path):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 2)  # Binary classification
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()

    predicted_class = "Cat" if probabilities[0] > probabilities[1] else "Dog"

    print(f"Predicted probabilities: {probabilities}")
    print(f"Predicted class: {predicted_class}")
    return predicted_class


# Main Execution for Training
if __name__ == "__main__":
    # #Q1
    # if not os.path.exists("mnist_vgg16_log.pth") or not os.path.exists("best_mnist_vgg16.pth"):
    #     print("Training VGG16 for Q1...")
    #     train_vgg16_and_save_logs()
    # else:
    #     print("Logs and model already exist. No training required.")
    
    #Q2
    dataset_path = "../Q2_Dataset/dataset/"
    if not os.path.exists("model/resnet50_no_erasing.pth"):
        print("Training ResNet50 without Random-Erasing...")
        train_resnet50("model/resnet50_no_erasing.pth", dataset_path, epochs=10, use_random_erasing=False)

    if not os.path.exists("model/resnet50_with_erasing.pth"):
        print("Training ResNet50 with Random-Erasing...")
        train_resnet50("model/resnet50_with_erasing.pth", dataset_path, epochs=10, use_random_erasing=True)


#Q2 V2
# Epoch 1/10, Train Acc: 0.9623, Val Acc: 0.9639
# Epoch 2/10, Train Acc: 0.9749, Val Acc: 0.9767
# Epoch 3/10, Train Acc: 0.9818, Val Acc: 0.9667
# Epoch 4/10, Train Acc: 0.9828, Val Acc: 0.9667
# Epoch 5/10, Train Acc: 0.9863, Val Acc: 0.9800
# Epoch 6/10, Train Acc: 0.9850, Val Acc: 0.9783
# Epoch 7/10, Train Acc: 0.9862, Val Acc: 0.9683
# Epoch 8/10, Train Acc: 0.9863, Val Acc: 0.9678
# Epoch 9/10, Train Acc: 0.9893, Val Acc: 0.9672
# Epoch 10/10, Train Acc: 0.9903, Val Acc: 0.9650
# Training completed. Best Validation Accuracy: 0.9800
# Training ResNet50 with Random-Erasing...
# Epoch 1/10, Train Acc: 0.9477, Val Acc: 0.9722
# Epoch 2/10, Train Acc: 0.9654, Val Acc: 0.9822
# Epoch 3/10, Train Acc: 0.9683, Val Acc: 0.9756
# Epoch 4/10, Train Acc: 0.9743, Val Acc: 0.9611
# Epoch 6/10, Train Acc: 0.9765, Val Acc: 0.9644
# Epoch 7/10, Train Acc: 0.9780, Val Acc: 0.9711
# Epoch 8/10, Train Acc: 0.9799, Val Acc: 0.9467
# Epoch 9/10, Train Acc: 0.9833, Val Acc: 0.9667
# Epoch 10/10, Train Acc: 0.9782, Val Acc: 0.9722
# Training completed. Best Validation Accuracy: 0.9822