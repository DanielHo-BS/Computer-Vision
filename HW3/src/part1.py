import numpy as np
import cv2
from utils import solve_homography, warping


def transform(img, canvas, corners, direction):
    """
    given a source image, a target canvas and the indicated corners, warp the source image to the target canvas
    :param img: input source image
    :param canvas: input canvas image
    :param corners: shape (4,2) numpy array, representing the four image corner (x, y) pairs
    :return: warped output image
    """
    h, w, ch = img.shape
    x = np.array([[0, 0],
                  [w, 0],
                  [w, h],
                  [0, h]
                  ])
    H = solve_homography(x, corners)
    if direction=='f':
        return  warping(img, canvas, H, 0, h, 0, w, direction=direction)
    
    elif direction=='b':
        xmin, ymin = np.min(corners, axis=0)
        xmax, ymax = np.max(corners, axis=0)
        return  warping(img, canvas, H, ymin, ymax, xmin, xmax, direction=direction)

    else:
        print("Wrong Direction!!")
        return None
        

if __name__ == "__main__":

    # ================== Part 1: Homography Estimation ========================
    canvas = cv2.imread('../resource/times.jpg')

    # TODO: 1.you can use whatever images you like, include these images in the img directory
    img1 = cv2.imread('../resource/img1.jpg')
    img2 = cv2.imread('../resource/img2.jpg')
    img3 = cv2.imread('../resource/img3.jpg')
    img4 = cv2.imread('../resource/img4.jpg')
    img5 = cv2.imread('../resource/img5.png')

    canvas_corners1 = np.array([[749, 521], [883, 525], [883, 750], [750, 750]])
    canvas_corners2 = np.array([[1395, 511], [1564, 434], [1573, 1013], [1402, 1012]])
    canvas_corners3 = np.array([[113, 185], [224, 268], [208, 519], [97, 474]])
    canvas_corners4 = np.array([[116, 632], [260, 684], [222, 956], [66, 949]])
    canvas_corners5 = np.array([[725, 62], [893, 62], [893, 191], [724, 192]])

    # TODO: 2. implement the transform function
    direction = 'f' # forword
    output1 = transform(img1, canvas, canvas_corners1, direction)
    output1 = transform(img2, output1, canvas_corners2, direction)
    output1 = transform(img3, output1, canvas_corners3, direction)
    output1 = transform(img4, output1, canvas_corners4, direction)
    output1 = transform(img5, output1, canvas_corners5, direction)

    cv2.imwrite('output1_1.png', output1)

    direction = 'b' # backward
    output2 = transform(img1, canvas, canvas_corners1, direction)
    output2 = transform(img2, output2, canvas_corners2, direction)
    output2 = transform(img3, output2, canvas_corners3, direction)
    output2 = transform(img4, output2, canvas_corners4, direction)
    output2 = transform(img5, output2, canvas_corners5, direction)

    cv2.imwrite('output1_2.png', output2)