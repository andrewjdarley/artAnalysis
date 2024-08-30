'''
So the whole point of this project is that Dr. Wingate's friend wants to analyze the works of caravaggio and then expand that. The 
features she's interested in particular are the lightings on the paintings. So entropy should show places wehre lighting changes.
'''


from PIL import Image
import numpy as np
import cv2, os
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

baroque_paths = os.listdir('dataset/wikiart/Baroque')

for path in baroque_paths:
    image_path = f"dataset/wikiart/Baroque/{path}"

    clss = path[:5]

    image = cv2.imread(image_path)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_image = pil_image.resize((400, int(pil_image.height * (400 / pil_image.width))))
    frame = np.array(pil_image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    entropy_image = entropy(gray_image, disk(5))

    entropy_normalized = img_as_ubyte(entropy_image / np.max(entropy_image))
    entropy_colored = cv2.cvtColor(entropy_normalized, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Original and Entropy Image", np.hstack([frame, entropy_colored]))
    cv2.waitKey(0)

    cv2.destroyAllWindows()
