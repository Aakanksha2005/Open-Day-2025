%pip install scikit-image

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import random
import os

def load_image(path):
    """Loads an image in color mode."""
    return cv2.imread(path, cv2.IMREAD_COLOR)  # Load in color mode

def calculate_similarity(img1, img2):
    """Computes Structural Similarity Index (SSIM) between two images."""
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1_resized = cv2.resize(img1_gray, (img2_gray.shape[1], img2_gray.shape[0]))  # Resize to match dimensions
    similarity_index = ssim(img1_resized, img2_gray)
    return similarity_index * 100  # Convert to percentage

def capture_from_camera():
    """Captures an image from the camera and displays emoji in color."""
    cap = cv2.VideoCapture(0)  # 0 usually refers to the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            cap.release()
            cv2.destroyAllWindows()
            return None

        # Flip the frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)

        # Resize the emoji
        emoji_resized = cv2.resize(original, (100, 100))  # Resize to fit the corner

        # Place the emoji in the top-left corner
        if frame.shape[0] > 100 and frame.shape[1] > 100:
            frame[10:110, 10:110] = emoji_resized  # Place emoji in top-left corner
        else:
            print("Warning: Frame is too small for emoji, skipping emoji display.")

        cv2.putText(frame, "Press ENTER to capture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Recreate the Emoji", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            captured_frame = frame.copy()  # Save the captured frame
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_frame  # Return captured frame in color mode

def choose_random_emoji(emoji_folder):
    """Chooses a random emoji from the specified folder."""
    emoji_files = [f for f in os.listdir(emoji_folder) if os.path.isfile(os.path.join(emoji_folder, f))]
    if not emoji_files:
        print("Error: No emoji images found in the folder.")
        return None

    chosen_file = random.choice(emoji_files)
    return os.path.join(emoji_folder, chosen_file)

def main():
    emoji_folder = "emojis"  # Folder containing emoji images. Create this folder and put your emoji images there.
    original_path = choose_random_emoji(emoji_folder)
    if original_path is None:
        return

    global original
    original = load_image(original_path)
    if original is None:
        print(f"Error: Could not load emoji image: {original_path}")
        return

    print(f"Recreate this emoji: {os.path.basename(original_path)}")  # Print the emoji file name

    recreated = capture_from_camera()
    if recreated is None:
        return

    match_percentage = calculate_similarity(original, recreated)
    if match_percentage < 60:
        match_percentage+=30
    elif  match_percentage < 70:
        match_percentage+=20 
    elif  match_percentage < 80:
        match_percentage+=10
    else:
        match_percentage+=0     
    
        

    # Display the final result
    cv2.putText(recreated, f"Match: {match_percentage:.2f}%", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Result", recreated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
