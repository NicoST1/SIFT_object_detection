import os

import cv2
import numpy as np
from generate_data import generate_test_data


def read_training_dataset(dir, imread_mode=cv2.IMREAD_GRAYSCALE):
    print(f'Reading training dataset: {dir}')
    return [(
        cv2.imread(dir + 'png/' + path, imread_mode),
        path
    ) for path in os.listdir(dir + 'png/')]

# just for testing that bb coords during test data generation are correct
def draw_oriented_bbox(image, top_left, bottom_right, angle, color=(0, 255, 0), thickness=2):
    # Compute the center, width, and height of the bounding box
    center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
    width = abs(top_left[0] - bottom_right[0])
    height = abs(top_left[1] - bottom_right[1])

    # Create a rectangle centered at the origin with the computed width and height
    rect = np.array([[-width // 2, -height // 2],
                     [width // 2, -height // 2],
                     [width // 2, height // 2],
                     [-width // 2, height // 2]], dtype=np.float32)

    # Compute the rotation matrix and rotate the rectangle
    angle_rad = np.radians(angle)
    rot_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]], dtype=np.float32)
    rotated_rect = np.dot(rect, rot_mat)

    # Translate the rotated rectangle to the correct position
    translated_rect = rotated_rect + np.array(center, dtype=np.float32)

    # Draw the oriented bounding box
    pts = translated_rect.astype(np.int32)
    for i in range(4):
        cv2.line(image, tuple(pts[i]), tuple(pts[(i+1)%4]), color, thickness)
    return image

training_data_dir = 'Task3/Task2Dataset/Training/'
training_paths_and_images = read_training_dataset(training_data_dir, cv2.IMREAD_COLOR)

draw_true_bbs = False
print('Generating new test data...')
for img, all_objects in generate_test_data(10, training_paths_and_images):
    print('All Data for image:', all_objects)
    for object_details in all_objects:
        name, top_left, bottom_right, angle = object_details
        if draw_true_bbs:
            img = draw_oriented_bbox(img, top_left, bottom_right, angle)

    partial_object_details = [obj[:3] for obj in all_objects]
    print('Data available for prediction:', partial_object_details)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
