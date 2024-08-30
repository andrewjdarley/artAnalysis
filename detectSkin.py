from PIL import Image
import numpy as np
import cv2, os

# Hardcoded path to the input image
for path in os.listdir('dataset/wikiart/Baroque'):
    if str(path).startswith('carav'):
        image_path = f"dataset/wikiart/Baroque/{path}"

        # Paths to save the original and output images
        original_output_path = "original_image.jpg"
        masked_output_path = "masked_image.jpg"

        # Adjust the HSV range for better skin detection
        lower = np.array([0, 20, 70], dtype="uint8")
        upper = np.array([25, 255, 255], dtype="uint8")

        image = cv2.imread(image_path)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_image = pil_image.resize((400, int(pil_image.height * (400 / pil_image.width))))
        frame = np.array(pil_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)

        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(frame, frame, mask=skinMask)

        # Save the original image
        cv2.imwrite(original_output_path, frame)

        # Save the masked image
        cv2.imwrite(masked_output_path, skin)

        # Display the images
        cv2.imshow("Original and Masked Image", np.hstack([frame, skin]))
        cv2.waitKey(0)

        cv2.destroyAllWindows()
