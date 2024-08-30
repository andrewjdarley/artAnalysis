from PIL import Image
import numpy as np
import cv2, os
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from random import sample
from alive_progress import alive_bar

'''
['Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 'Art_Nouveau_Modern', 'Baroque', 'Color_Field_Painting', 
'Contemporary_Realism', 'Cubism', 'Early_Renaissance', 'Expressionism', 'Fauvism', 'High_Renaissance', 'Impressionism', 
'Mannerism_Late_Renaissance', 'Minimalism', 'Naive_Art_Primitivism', 'New_Realism', 'Northern_Renaissance', 'Pointillism', 
'Pop_Art', 'Post_Impressionism', 'Realism', 'Rococo', 'Romanticism', 'Symbolism', 'Synthetic_Cubism', 'Ukiyo_e']
'''

styles = [
    # 'Baroque', 
    'High_Renaissance',
    'Northern_Renaissance',
    'Mannerism_Late_Renaissance',
    'Early_Renaissance',
    'Realism',
    'New_Realism',
    'Romanticism',
    'Rococo'
]
for genre in styles:
    baroque_paths = os.listdir(f'dataset/wikiart/{genre}')

    snippet_size = 224
    num_samples = 7
    max_samples = 12
    min_distance = 120
    max_iterations = num_samples * 4
    percentile = 75  # i.e., 75 is 75th percentile, or will sample from top 25 percent entropy
    wallpad = 100

    if not os.path.exists('snippets'):
        os.makedirs('snippets')

    def is_far_enough(point, selected_points, min_distance):
        """Check if a point is at least min_distance away from all selected points."""
        for sel_point in selected_points:
            if np.linalg.norm(np.array(point) - np.array(sel_point)) < min_distance:
                return False
        return True

    with alive_bar(len(baroque_paths)) as bar:
        for path in baroque_paths:
            try:
                bar()
                image_path = f"dataset/wikiart/{genre}/{path}"

                clss = path.split('_')[0]

                image = cv2.imread(image_path)

                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Calculate entropy
                entropy_image = entropy(gray_image, disk(5))
                entropy_normalized = img_as_ubyte(entropy_image / np.max(entropy_image))

                # Determine the percentile of entropy values
                threshold_value = np.percentile(entropy_normalized, percentile)

                # Get coordinates of all points above the percentile
                high_entropy_coords = np.column_stack(np.where(entropy_normalized >= threshold_value))

                # Filter out points that are too close to the edges (wallpad)
                high_entropy_coords = [
                    coord for coord in high_entropy_coords 
                    if wallpad <= coord[0] < image.shape[0] - wallpad and wallpad <= coord[1] < image.shape[1] - wallpad
                ]

                # Randomly sample from these points, don't let them be within min distance
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
                    y_end = min(y_start + snippet_size, image.shape[0])
                    x_end = min(x_start + snippet_size, image.shape[1])

                    snippet = image[y_start:y_end, x_start:x_end]

                    # Apply reflection padding if the snippet is smaller than the desired size
                    if snippet.shape[0] < snippet_size or snippet.shape[1] < snippet_size:
                        top_pad = (snippet_size - snippet
                                   .shape[0]) // 2
                        bottom_pad = snippet_size - snippet.shape[0] - top_pad
                        left_pad = (snippet_size - snippet.shape[1]) // 2
                        right_pad = snippet_size - snippet.shape[1] - left_pad

                        padded_snippet = cv2.copyMakeBorder(
                            snippet, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_REFLECT
                        )
                    else:
                        padded_snippet = snippet

                    os.makedirs(f'snippets/{clss}', exist_ok=True)
                    cv2.imwrite(f'snippets/{clss}/{path}_{i}.jpg', padded_snippet)
            except:
                pass
