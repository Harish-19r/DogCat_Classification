import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import DogCatCNN

# -----------------------------
# 1. Select training device
# -----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Training device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# 2. Data transformations
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

# -----------------------------
# 3. Dataset path
# -----------------------------
dataset_path = r"C:\Users\Hp\OneDrive\Documents\PetImages"

# Load dataset
full_dataset = datasets.ImageFolder(
    dataset_path,
    transform=train_transform
)

print("Classes:", full_dataset.classes)
print("Total images:", len(full_dataset))

# -----------------------------
# 4. Train / validation split
# -----------------------------
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

# Give validation images the validation transform
val_dataset.dataset.transform = val_transform

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# -----------------------------
# 5. DataLoaders
# -----------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# -----------------------------
# 6. Create model
# -----------------------------
model = DogCatCNN().to(device)

print("Model created successfully!")

# -----------------------------
# 7. Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

# -----------------------------
# 8. Training settings
# -----------------------------
num_epochs = 15
best_val_accuracy = 0.0
patience = 3
epochs_without_improvement = 0

print()
print("Starting training...")
print()

# -----------------------------
# 9. Training loop
# -----------------------------
for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_total += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss_total / len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    # -----------------------------
    # Save best model
    # -----------------------------
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "dog_cat_model_best.pth"
        )

        print("  ✓ Best model saved!")

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

        print(
            f"  No improvement "
            f"({epochs_without_improvement}/{patience})"
        )

    # -----------------------------
    # Early stopping
    # -----------------------------
    if epochs_without_improvement >= patience:

        print()
        print("Early stopping triggered.")
        break

print()
print("Training complete!")
print("Best validation accuracy:", f"{best_val_accuracy:.2f}%")
print("Best model saved as: dog_cat_model_best.pth")