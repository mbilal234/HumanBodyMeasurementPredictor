import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("final_model.h5")

def preprocess_image(image):
    # Resize image to (64, 64)
    resized_image = cv2.resize(image, (128, 128))
    
    # Convert to RGBA format
    rgba_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGBA)

    # Convert to float32 and normalize
    normalized_image = rgba_image.astype(np.float32) / 255.0

    # Add batch dimension
    preprocessed_image = np.expand_dims(normalized_image, axis=0)

    return preprocessed_image

# Function to capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    
    # Get the video width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Open a new window
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam', width, height)
    
    # Display countdown on the video feed
    for i in range(10, 0, -1):
        ret, frame = cap.read()  # Capture frame from webcam
        cv2.putText(frame, str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)
        cv2.waitKey(1000)  # Wait for 1 second
    
    # Capture frame after countdown
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    
    if ret:
        print("Image captured successfully.")
        return frame
    else:
        print("Failed to capture image.")
        return None

# Function to make predictions
def make_predictions(image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    return predictions

# Main function
def main():
    # Capture image from webcam
    image = capture_image()

    if image is not None:
        # Make predictions
        predictions = make_predictions(image)

        # Display predictions in a dialog box
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        # Convert predictions to a string
        predictions_str = "\n".join([f"Measurement {i+1}: {prediction}" for i, prediction in enumerate(predictions[0])])
        
        # Display predictions in a message box
        messagebox.showinfo("Your measurements are:", predictions_str)

if __name__ == "__main__":
    main()
