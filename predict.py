import torch
from torchvision import transforms
from PIL import Image
from model import DogCatCNN

# Use NVIDIA GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# Load trained model
model = DogCatCNN()
model.load_state_dict(
    torch.load("dog_cat_model_best.pth", map_location=device)
)
model = model.to(device)
model.eval()

# Same preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

classes = ["Cat", "Dog"]

# Test both images
for filename in ["dogg.jpg", "test_dog6.jpg", "test_cat6.jpg", "test_dog5.jpg", "test_cat5.jpg", "test_dog4.jpg", "test_cat4.jpg", "test_dog3.jpg", "test_cat3.jpg", "test_dog2.jpg", "test_cat2.jpg", "test_dog1.jpg", "test_cat1.jpg", "dog1.jpg", "cat1.jpg", "dog2.jpg", "cat2.jpg", "dog3.jpg", "cat3.jpg", "dog4.jpg", "cat4.jpg", "dog5.jpg", "cat5.jpg", ]:

    image = Image.open(filename).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted = torch.argmax(output, dim=1).item()

    print()
    print("Image:", filename)
    print("Prediction:", classes[predicted])
    print(
        "Confidence:",
        f"{probabilities[0][predicted].item() * 100:.2f}%"
    )

print()
print("Testing complete!")