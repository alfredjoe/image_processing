import cv2
import numpy as np
import os
from flask import Flask, render_template

app = Flask(__name__)

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return {}

    # Get image size & shape
    height, width, channels = image.shape
    dimensions = f"{width}x{height}"

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_path = "static/gray_image.jpg"
    cv2.imwrite(gray_path, gray_image)

    # Convert to binary
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    binary_path = "static/binary_image.jpg"
    cv2.imwrite(binary_path, binary_image)

    # Scale down
    scaled_image = cv2.resize(image, (width // 2, height // 2))
    scaled_path = "static/scaled_image.jpg"
    cv2.imwrite(scaled_path, scaled_image)

    # Denoise
    denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    denoised_path = "static/denoised_image.jpg"
    cv2.imwrite(denoised_path, denoised_image)

    return {
        "input": image_path,
        "gray": gray_path,
        "binary": binary_path,
        "scaled": scaled_path,
        "denoised": denoised_path,
    }, dimensions, channels

@app.route("/")
def display_images():
    image_path = "static/input_image.jpg"  # Ensure image exists in 'static/'
    images, dimensions, channels = process_image(image_path)
    return render_template("index.html", images=images, dimensions=dimensions, channels=channels)

if __name__ == "__main__":
    app.run(debug=True)
