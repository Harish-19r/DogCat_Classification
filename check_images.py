from PIL import Image
import os

dataset_path = r"C:\Users\Hp\OneDrive\Documents\PetImages"

bad_images = []
total = 0

for category in ["Cat", "Dog"]:
    folder = os.path.join(dataset_path, category)

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if os.path.isfile(filepath):
            total += 1

            try:
                with Image.open(filepath) as img:
                    img.verify()
            except Exception:
                bad_images.append(filepath)

print("Total files checked:", total)
print("Bad images found:", len(bad_images))

if bad_images:
    print("\nBad images:")
    for image in bad_images:
        print(image)