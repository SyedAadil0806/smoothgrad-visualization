import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


# ImageNet normalization values used by ResNet18
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


def choose_image(folder_path): #Helps choose the image
    image_files = []

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_files.append(file)

    image_files.sort()

    if len(image_files) == 0:
        print("No images found in test_images folder.")
        exit()

    print("\nImages found:\n")

    for i, file in enumerate(image_files, start=1):
        print(str(i) + ". " + file)

    while True:
        try:
            choice = int(input("\nEnter the image number you want to use: "))

            if choice >= 1 and choice <= len(image_files):
                selected_image = image_files[choice - 1]
                return os.path.join(folder_path, selected_image)
            else:
                print("Invalid number. Please choose from the list.")

        except ValueError:
            print("Please enter a valid number.")


def load_image(image_path): #loads the image for processing
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    input_tensor.requires_grad = True

    return image, input_tensor


def get_class_name(class_names, class_index): #Converts the class number into a readable name
    class_name = class_names[class_index]
    class_name = class_name.replace("_", " ")
    return class_name.title()


def get_normal_gradient(model, input_tensor): #Creates the normal gradient sensitivity map
    model.zero_grad() 

    output = model(input_tensor)
    predicted_class = output.argmax().item()

    score = output[0, predicted_class]
    score.backward()

    gradient = input_tensor.grad.abs()
    gradient = gradient.squeeze().mean(dim=0)

    return gradient.detach().numpy(), predicted_class


def get_smoothgrad(model, input_tensor, target_class, samples=50, noise_level=0.15): #Implements SmoothGrad
    total_gradient = torch.zeros_like(input_tensor)

    # FIX 1: Scale noise to input range so noise_level is a true fraction of
    # the input's value range, matching the paper's definition of
    # sigma / (x_max - x_min).
    input_range = input_tensor.max() - input_tensor.min()

    for i in range(samples):
        noise = torch.randn_like(input_tensor) * noise_level * input_range

        noisy_image = (input_tensor + noise).detach()
        noisy_image.requires_grad_(True)

        model.zero_grad()

        output = model(noisy_image)

        score = output[0, target_class]
        score.backward()

        total_gradient += noisy_image.grad

    smooth_gradient = total_gradient / samples
    smooth_gradient = smooth_gradient.abs()
    smooth_gradient = smooth_gradient.squeeze().mean(dim=0)

    return smooth_gradient.detach().numpy()


def normalize_map(gradient_map): #Makes heatmap values between 0 and 1
    # FIX 2: Cap at 99th percentile before normalizing, as recommended by the
    # paper to prevent a few outlier pixels from washing out the entire map.
    p99 = np.percentile(gradient_map, 99)
    gradient_map = np.clip(gradient_map, 0, p99)

    gradient_map = gradient_map - gradient_map.min()

    if gradient_map.max() != 0:
        gradient_map = gradient_map / gradient_map.max()

    return gradient_map


def create_overlay(original_image, gradient_map): #Creates blended visualization
    original_resized = original_image.resize((224, 224))
    original_array = np.array(original_resized) / 255.0

    gradient_map = normalize_map(gradient_map)

    heatmap = plt.cm.hot(gradient_map)[:, :, :3]

    overlay = 0.55 * original_array + 0.45 * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay


def clean_filename(class_name):
    name = class_name.lower()
    name = name.replace(" ", "-")
    name = re.sub(r"[^a-z0-9\-]", "", name)
    return name


def get_next_output_name(class_name, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    clean_name = clean_filename(class_name)

    existing_numbers = []

    for file in os.listdir(output_folder):
        if file.startswith(clean_name) and file.endswith(".jpg"):
            number_part = file.replace(clean_name, "").replace(".jpg", "")

            if number_part.isdigit():
                existing_numbers.append(int(number_part))

    if len(existing_numbers) == 0:
        next_number = 1
    else:
        next_number = max(existing_numbers) + 1

    file_name = clean_name + str(next_number) + ".jpg"
    return os.path.join(output_folder, file_name)


def show_and_save_results(original_image, normal_gradient, smooth_gradient, class_name, output_folder):
    normal_gradient = normalize_map(normal_gradient)
    smooth_gradient = normalize_map(smooth_gradient)

    overlay = create_overlay(original_image, smooth_gradient)

    save_path = get_next_output_name(class_name, output_folder)

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(original_image.resize((224, 224)))
    plt.title("Original Image", fontsize=12)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(normal_gradient, cmap="hot")
    plt.title("Normal Gradient", fontsize=12)
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(smooth_gradient, cmap="hot")
    plt.title("SmoothGrad", fontsize=12)
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(overlay)
    plt.title("SmoothGrad Overlay", fontsize=12)
    plt.axis("off")

    plt.suptitle("Predicted Class: " + class_name, fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.82, wspace=0.18)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return save_path