import cv2
import numpy as np

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    
    # Print the size and shape of the image
    height, width, channels = image.shape
    print(f"Image Dimensions: {width}x{height}")
    print(f"Number of Channels: {channels}")
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray_image.jpg", gray_image)
    
    # Convert the image to binary format
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite("binary_image.jpg", binary_image)
    
    # Scale down the image (reduce size by 50%)
    scaled_image = cv2.resize(image, (width // 2, height // 2))
    cv2.imwrite("scaled_image.jpg", scaled_image)
    
    # Remove noise using Gaussian Blur
    denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    cv2.imwrite("denoised_image.jpg", denoised_image)
    
    print("Image processing completed. Processed images are saved.")

# Provide the image path here
image_path = "input_image.jpg"  # Replace with your actual image path
process_image(image_path)
