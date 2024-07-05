import cv2
import numpy as np
from PIL import Image
import urllib.request

# Function to capture image from webcam
def capture_webcam_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame

# Function to download OpenPose model files from GitHub
def download_model_files():
    prototxt_url = "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt"
    caffemodel_url = "https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/coco/pose_iter_440000.caffemodel?raw=true"
    
    urllib.request.urlretrieve(prototxt_url, "pose_deploy_linevec.prototxt")
    urllib.request.urlretrieve(caffemodel_url, "pose_iter_440000.caffemodel")

# Function to process image using OpenPose for segmentation
def process_image(image):
    # Download model files if not already downloaded
    download_model_files()

    # Load OpenPose model
    net = cv2.dnn.readNetFromTensorflow("pose_deploy_linevec.prototxt", "pose_iter_440000.caffemodel")

    # Specify target size for the input image to OpenPose
    target_size = (368, 368)

    # Resize image to target size
    resized_image = cv2.resize(image, target_size)

    # Prepare the input blob for OpenPose
    blob = cv2.dnn.blobFromImage(resized_image, 1.0 / 255, target_size, (0, 0, 0), swapRB=False, crop=False)

    # Set the prepared blob as input to the network
    net.setInput(blob)

    # Run forward pass through the network to perform inference
    output = net.forward()

    # Reshape the output to have 5 dimensions
    output = output.squeeze()

    # Extract the keypoint heatmaps from the output
    keypoint_heatmaps = output[0:19, :, :]

    # Compute the global maximum of the heatmaps
    max_val = np.max(keypoint_heatmaps)

    # Threshold to get binary mask of the person
    _, person_mask = cv2.threshold(keypoint_heatmaps[0], max_val * 0.1, 255, cv2.THRESH_BINARY)

    # Convert mask to 3 channels (for compatibility with PIL)
    person_mask = cv2.merge((person_mask, person_mask, person_mask))

    # Resize mask to original image size
    person_mask = cv2.resize(person_mask, (image.shape[1], image.shape[0]))

    # Create white silhouette of person on black background
    white_silhouette = np.where(person_mask == 255, 255, 0)

    return white_silhouette

# Function to save image
def save_image(image, filename):
    cv2.imwrite(filename, image)

def main():
    # Capture image from webcam
    image = capture_webcam_image()

    # Process image using OpenPose
    processed_image = process_image(image)

    # Save processed image
    save_image(processed_image, "silhouette.png")

if __name__ == "__main__":
    main()
