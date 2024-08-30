from PIL import Image
import numpy as np
import cv2, os
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from random import sample

GENRE = 'High_Renaissance'
baroque_paths = os.listdir(f'dataset/wikiart/{GENRE}')

snippet_size = 224
num_samples = 7
max_samples = 12
min_distance = 120
max_iterations = num_samples * 4
percentile = 50 # ie: 75 is 75th percentile, or will sample from top 25 percent entropy

if not os.path.exists('snippets'):
    os.makedirs('snippets')

def is_far_enough(point, selected_points, min_distance):
    """Check if a point is at least min_distance away from all selected points."""
    for sel_point in selected_points:
        if np.linalg.norm(np.array(point) - np.array(sel_point)) < min_distance:
            return False
    return True

for path in baroque_paths:
    image_path = f"dataset/wikiart/{GENRE}/{path}"

    clss = path[:5]

    image = cv2.imread(image_path)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_image = pil_image.resize((400, int(pil_image.height * (400 / pil_image.width))))
    frame = np.array(pil_image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate entropy
    entropy_image = entropy(gray_image, disk(5))
    entropy_normalized = img_as_ubyte(entropy_image / np.max(entropy_image))

    # Determine the percentile of entropy values
    threshold_value = np.percentile(entropy_normalized, percentile)

    # Get coordinates of all points above the percentile
    high_entropy_coords = np.column_stack(np.where(entropy_normalized >= threshold_value))

    # Randomly sample from these points, ensuring they are not within min_distance of each other
    selected_coords = []
    iteration_count = 0
    while len(selected_coords) < min(num_samples, max_samples) and len(high_entropy_coords) > 0:
        candidate = sample(list(high_entropy_coords), 1)[0]
        if is_far_enough(candidate, selected_coords, min_distance):
            selected_coords.append(candidate)
            # Remove the selected candidate from high_entropy_coords
            high_entropy_coords = np.delete(high_entropy_coords, np.where((high_entropy_coords == candidate).all(axis=1)), axis=0)
        iteration_count += 1
        if iteration_count >= max_iterations:
            break

    # Extract and save snippets
    for i, (y, x) in enumerate(selected_coords):
        y_start = max(y - snippet_size // 2, 0)
        x_start = max(x - snippet_size // 2, 0)

        # Ensure the snippet is within bounds
        y_end = min(y_start + snippet_size, frame.shape[0])
        x_end = min(x_start + snippet_size, frame.shape[1])

        snippet = frame[y_start:y_end, x_start:x_end]

        # Apply reflection padding if the snippet is smaller than the desired size
        if snippet.shape[0] < snippet_size or snippet.shape[1] < snippet_size:
            top_pad = (snippet_size - snippet.shape[0]) // 2
            bottom_pad = snippet_size - snippet.shape[0] - top_pad
            left_pad = (snippet_size - snippet.shape[1]) // 2
            right_pad = snippet_size - snippet.shape[1] - left_pad

            padded_snippet = cv2.copyMakeBorder(
                snippet, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_REFLECT
            )
        else:
            padded_snippet = snippet

        # os.makedirs(f'snippets/{clss}', exist_ok=True)
        # cv2.imwrite(f'snippets/{clss}/{path}_{i}.jpg', padded_snippet)

    # Display the original and entropy images with points marked
    entropy_colored = cv2.cvtColor(entropy_normalized, cv2.COLOR_GRAY2BGR)
    for y, x in selected_coords:
        cv2.circle(entropy_colored, (x, y), 10, (0, 255, 0), 2)

    cv2.imshow("Original and Entropy Image with Selected Points", np.hstack([frame, entropy_colored]))
    cv2.waitKey(0)

    cv2.destroyAllWindows()
