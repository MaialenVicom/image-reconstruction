import cv2
import numpy as np
import random


def add_uneven_illumination(image, intensity=None, center=None, sigma=None):
    if intensity is None:
        intensity = random.uniform(0.5, 1.5)
    if center is None:
        h, w = image.shape[:2]
        center = (
            random.randint(0, w),
            random.randint(0, h)
        )
    if sigma is None:
        h, w = image.shape[:2]
        max_dim = max(h, w)
        sigma = random.uniform(max_dim * 0.1, max_dim * 0.3)  # Adjustable spread of light

    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]

    # Create Gaussian illumination
    gaussian = np.exp(-((X - center[0]) ** 2 + (Y - center[1]) ** 2) / (2 * sigma ** 2))

    # Normalize and adjust intensity
    gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
    illumination = 1 + (gaussian * intensity)

    # Ensure minimum illumination isn't too dark
    min_illumination = 0.6
    illumination = min_illumination + (1 - min_illumination) * illumination

    # Apply soft illumination
    result = image.copy().astype(float)
    result = result * illumination[:, :, np.newaxis]
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def add_random_shadows(image, num_shadows=None, shadow_intensity=None):
    if num_shadows is None:
        num_shadows = random.randint(1, 3)
    if shadow_intensity is None:
        shadow_intensity = random.uniform(0.4, 0.8)

    result = image.copy().astype(float)
    h, w = image.shape[:2]

    for _ in range(num_shadows):
        # Create gradient shadow
        gradient_type = random.choice(['linear', 'radial'])

        if gradient_type == 'linear':
            # Linear gradient shadow
            angle = random.uniform(0, 2 * np.pi)
            x = np.linspace(0, w - 1, w)
            y = np.linspace(0, h - 1, h)
            X, Y = np.meshgrid(x, y)

            # Create directional gradient
            gradient = (X * np.cos(angle) + Y * np.sin(angle)) / np.sqrt(w ** 2 + h ** 2)
            gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())

            # Smooth the gradient
            gradient = cv2.GaussianBlur(gradient, (0, 0), sigmaX=w // 8)

        else:
            # Radial gradient shadow
            center = (
                random.randint(-w // 2, w * 3 // 2),
                random.randint(-h // 2, h * 3 // 2)
            )
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

            # Normalize distances and create smooth falloff
            max_dist = np.sqrt(w ** 2 + h ** 2)
            gradient = dist_from_center / max_dist
            gradient = cv2.GaussianBlur(gradient, (0, 0), sigmaX=w // 8)

        # Normalize and adjust shadow intensity
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        shadow_mask = 1 - (gradient * (1 - shadow_intensity))

        # Apply shadow with soft transition
        result = result * shadow_mask[:, :, np.newaxis]

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def add_elliptical_reflections(image, num_reflections=None, intensity=None):
    if num_reflections is None:
        num_reflections = random.randint(1, 3)
    if intensity is None:
        intensity = random.uniform(0.3, 0.7)

    result = image.copy().astype(float)
    h, w = image.shape[:2]

    for _ in range(num_reflections):
        reflection_mask = np.zeros((h, w), dtype=np.float32)
        center = (random.randint(0, w), random.randint(0, h))
        major_axis = random.randint(w // 20, w // 5)
        minor_axis = random.randint(w // 30, major_axis)
        angle = random.uniform(0, 180)

        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(temp_mask, center, (major_axis, minor_axis),
                    angle, 0, 360, 255, -1)

        reflection_mask = temp_mask.astype(float) / 255.0
        sigma = random.uniform(5, 15)
        reflection_mask = cv2.GaussianBlur(reflection_mask, (0, 0), sigma)
        if reflection_mask.max() > 0:
            reflection_mask = reflection_mask / reflection_mask.max()

        reflection_color = np.array([random.uniform(0.8, 1.0) for _ in range(3)])
        for c in range(3):
            channel = result[:, :, c]
            reflection_contribution = (255 - channel) * reflection_mask * intensity * reflection_color[c]
            channel += reflection_contribution
            result[:, :, c] = channel

    return np.clip(result, 0, 255).astype(np.uint8)

def add_streak_reflections(image, num_reflections=None, intensity=None):
    if num_reflections is None:
        num_reflections = random.randint(1, 3)
    if intensity is None:
        intensity = random.uniform(0.3, 0.7)

    result = image.copy().astype(float)
    h, w = image.shape[:2]

    for _ in range(num_reflections):
        reflection_mask = np.zeros((h, w), dtype=np.float32)

        start_point = (random.randint(-w // 4, w * 5 // 4), random.randint(-h // 4, h * 5 // 4))
        control_point = (random.randint(-w // 4, w * 5 // 4), random.randint(-h // 4, h * 5 // 4))
        end_point = (random.randint(-w // 4, w * 5 // 4), random.randint(-h // 4, h * 5 // 4))

        pts = np.array([start_point, control_point, end_point], dtype=np.float32)
        t_vals = np.linspace(0, 1, 100)
        curve_points = np.array([
            (1 - t)**2 * pts[0] + 2 * (1 - t) * t * pts[1] + t**2 * pts[2]
            for t in t_vals
        ], dtype=np.int32)

        temp_mask = np.zeros((h, w), dtype=np.uint8)
        thickness = random.randint(5, 20)
        cv2.polylines(temp_mask, [curve_points], False, 255, thickness)

        reflection_mask = temp_mask.astype(float) / 255.0
        sigma = random.uniform(5, 15)
        reflection_mask = cv2.GaussianBlur(reflection_mask, (0, 0), sigma)
        if reflection_mask.max() > 0:
            reflection_mask = reflection_mask / reflection_mask.max()

        reflection_color = np.array([random.uniform(0.8, 1.0) for _ in range(3)])
        for c in range(3):
            channel = result[:, :, c]
            reflection_contribution = (255 - channel) * reflection_mask * intensity * reflection_color[c]
            channel += reflection_contribution
            result[:, :, c] = channel

    return np.clip(result, 0, 255).astype(np.uint8)


def add_window_info(image, title):
    """Add title text to the image"""
    result = image.copy()
    cv2.putText(result, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return result


def main():
    # Load your image
    image = cv2.imread('C:/DATA/sources/InnitiusVisualize/image-processing-pipeline-to-be-embedded/tests/data/0.jpg')
    if image is None:
        print("Error: Could not load image")
        return

    original_image = image.copy()

    # Create named windows and position them
    cv2.namedWindow('Original Image')
    cv2.namedWindow('Uneven Illumination')
    cv2.namedWindow('Random Shadows')
    cv2.namedWindow('Specular Reflections')

    # Position windows in a 2x2 grid
    screen_width = 1920
    screen_height = 1080
    window_width = screen_width // 2
    window_height = screen_height // 2

    # Position windows
    windows = {
        'Original Image': (0, 0),
        'Uneven Illumination': (window_width, 0),
        'Random Shadows': (0, window_height),
        'Specular Reflections': (window_width, window_height)
    }

    for window, pos in windows.items():
        cv2.moveWindow(window, pos[0], pos[1])

    while True:
        # Create augmented images
        illuminated_image = add_uneven_illumination(original_image)
        shadowed_image = add_random_shadows(original_image)
        specular_image = add_streak_reflections(original_image)
        specular_image2= add_elliptical_reflections(original_image)

        # Add titles to images
        original_with_text = add_window_info(original_image, "Original")
        illuminated_with_text = add_window_info(illuminated_image, "Uneven Illumination")
        shadowed_with_text = add_window_info(shadowed_image, "Random Shadows")
        specular_with_text = add_window_info(specular_image, "Specular Reflections")
        specular_with_text2 = add_window_info(specular_image2, "Specular Reflections 2")

        # Display all images
        cv2.imshow('Original Image', original_with_text)
        cv2.imshow('Uneven Illumination', illuminated_with_text)
        cv2.imshow('Random Shadows', shadowed_with_text)
        cv2.imshow('Specular Reflections', specular_with_text)
        cv2.imshow('Specular Reflections', specular_with_text2)
        # Wait for keyboard input
        key = cv2.waitKey(0) & 0xFF

        # Handle keyboard input
        if key == ord('q'):  # Quit
            break
        # Any other key will trigger new random generations

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
