import cv2
import numpy as np

def scale_dimensions(dimension, scale):
    return round(dimension/scale)


def get_orientated_bounding_box(contour, scale=10):
    min_rect = cv2.minAreaRect(contour)
    (center, size, angle) = min_rect

    # Scale the min_rect back to the original image size
    center = tuple(np.array(center) / scale)
    size = tuple(np.array(size) / scale)

    # Create a new min_rect with the scaled values
    scaled_min_rect = (center, size, angle)

    # Convert the min_rect to a 4-point bounding box
    box = cv2.boxPoints(scaled_min_rect)
    box = np.int0(box)
    return box


def add_border(x_min, y_min, x_max, y_max, width, height, border=1):
    x_min = max(x_min-border, 0)
    y_min = max(y_min-border, 0)
    x_max = min(x_max+border, height)
    y_max = min(y_max+border, width)
    return x_min, y_min, x_max, y_max

def get_bounding_boxes(image, scale=10, min_area=500):
    bounding_boxes = []
    width, height = image.shape[:2]
    # Resize the image by the scaling factor as it improves the accuracy of the contour detection
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # Binarise and remove noise from the image
    image = cv2.medianBlur(image, 25)
    image_res, image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((11, 11), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = [scale_dimensions(elem, scale) for elem in cv2.boundingRect(c)]
        if w * h < min_area:
            continue
        x_min, y_min, x_max, y_max = add_border(x, y, x+w, y+h, width, height)
        bounding_boxes.append(((x_min, y_min, x_max, y_max), get_orientated_bounding_box(c)))
    return bounding_boxes


def segment_icons(image):
    image_segments = []
    # get bounding boxes for the image
    bounding_boxes = get_bounding_boxes(image.copy())

    for (x_min, y_min, x_max, y_max), bbox in bounding_boxes:
        image_segment = image[y_min:y_max, x_min:x_max].copy()
        border = 5
        padded_image = cv2.copyMakeBorder(
            image_segment,
            top=border,
            bottom=border,
            left=border,
            right=border,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        image_segments.append((padded_image, bbox))
    return image_segments