import random

import cv2
import numpy as np


def generate_test_data(
    n_images,
    training_images_and_paths,
    test_img_size=(512, 512),
    noise_density=0.001,
    min_scale=0.06,
    max_scale= 0.17,
    min_gap=10,
    min_objects=3,
    max_objects=5,
    do_rotate_images=True
):
    assert min_scale <= max_scale
    assert min_objects <= max_objects

    test_images_data = []
    for _ in range(n_images):
        # Create a white image then add some noise as seen in original test data
        test_img = np.full(test_img_size + (3,), 255, dtype=np.uint8)
        add_noise_to_image(test_img, noise_density) # TODO: This needs tuning

        placed_objects = []
        placed_objects_info = []

        # Generate a random sample of unique features (no duplicates)
        num_objects = random.randint(min_objects, max_objects)
        selected_features = random.sample(training_images_and_paths, num_objects)

        for img, img_path in selected_features:
            # Rotate the image
            if do_rotate_images:
                angle = 360 * random.random()
                img = rotate_image(img, angle)

            # Scale the image
            scale = random.uniform(min_scale, max_scale)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

            # Get a suitable placement that does not overlap other features
            img_h, img_w, _ = img.shape
            while True:
                x = random.randint(0, test_img_size[1] - img_w)
                y = random.randint(0, test_img_size[0] - img_h)
                overlap = False
                for obj_x, obj_y, obj_w, obj_h in placed_objects:
                    if (x > obj_x - min_gap - img_w and x < obj_x + obj_w + min_gap and
                            y > obj_y - min_gap - img_h and y < obj_y + obj_h + min_gap):
                        overlap = True
                        break
                if not overlap:
                    break

            # Add the feature to the image
            mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            roi = test_img[y:y+img_h, x:x+img_w]
            img_fg = cv2.bitwise_and(img, img, mask=mask)
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            dst = cv2.add(img_fg, roi_bg)
            test_img[y:y+img_h, x:x+img_w] = dst

            # Append the details
            placed_objects.append((x, y, img_w, img_h))
            placed_objects_info.append((feature_name_from_path(img_path), (x, y), (x+img_w, y+img_h), angle))

        test_images_data.append((test_img, placed_objects_info))

    return test_images_data


def add_noise_to_image(img, noise_density, blur_ksize=(3, 3)):
    img_size = img.shape

    # Generate noise as semi-transparent black dots on a separate black image
    for _ in range(int(img_size[0] * img_size[1] * noise_density)):
        x = random.randint(0, img_size[1] - 1)
        y = random.randint(0, img_size[0] - 1)
        cv2.circle(img, (x, y), 0, (150, 150, 150), -1)

    # Blur the noise image
    img = cv2.blur(img, blur_ksize)
    return img

def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # bring image to new centre
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    # TODO: fix black border issue
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def feature_name_from_path(img_path):
    return img_path[img_path.find('-')+1:img_path.find('.png')]
