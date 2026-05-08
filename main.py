import os
import torchvision.models as models

from utils import (
    choose_image,
    load_image,
    get_class_name,
    get_normal_gradient,
    get_smoothgrad,
    show_and_save_results
)


def main():
    print("\nLoading pretrained ResNet18 model...")

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    class_names = weights.meta["categories"]

    image_path = choose_image("test_images")

    print("\nLoading image...")
    original_image, input_tensor = load_image(image_path)

    print("Generating normal gradient map...")
    normal_gradient, predicted_class_index = get_normal_gradient(model, input_tensor)

    class_name = get_class_name(class_names, predicted_class_index)

    print("Predicted Class:", class_name)

    print("Generating SmoothGrad map...")

    original_image, input_tensor = load_image(image_path)

    smooth_gradient = get_smoothgrad(
        model,
        input_tensor,
        predicted_class_index,
        samples=50,
        noise_level=0.15
    )

    print("Saving final output...")

    saved_file = show_and_save_results(
        original_image,
        normal_gradient,
        smooth_gradient,
        class_name,
        "outputs"
    )

    print("\nDone.")
    print("File saved as:", saved_file)


if __name__ == "__main__":
    main()