import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


cv2.setRNGSeed(0)

# Define constants
n_workers = cv2.getNumberOfCPUs()
NORMAL = '\u001b[0m'
RED = '\u001b[31m'
GREEN = '\u001b[32m'


def filter_covered_boxes(bounding_boxes):
    filtered_boxes = []

    for i1, ((x1, y1, x2, y2), bbx1) in enumerate(bounding_boxes):
        larger = True
        for i2, ((x3, y3, x4, y4), _) in enumerate(bounding_boxes):
            if i1 == i2:
                continue
            if x3 <= x1 and x4 >= x2 and y3 <= y1 and y4 >= y2:
                larger = False
                break
        if larger:
            filtered_boxes.append(((x1, y1, x2, y2), bbx1))
    return filtered_boxes


def scale_dimensions(dimension, scale):
    return round(dimension/scale)


def add_border(x_min, y_min, x_max, y_max, width, height, border=1):
    x_min = max(x_min-border, 0)
    y_min = max(y_min-border, 0)
    x_max = min(x_max+border, height)
    y_max = min(y_max+border, width)
    return x_min, y_min, x_max, y_max


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


def get_bounding_boxes(image, scale=10, min_area=500):
    bounding_boxes = []
    width, height = image.shape[:2]
    # Resize the image by the scaling factor as it improves the accuracy of the contour detection
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # Binarise and remove noise from the image
    image = cv2.medianBlur(image, 25)
    _, image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((11, 11), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = [scale_dimensions(elem, scale) for elem in cv2.boundingRect(c)]
        # Ignore small contours that are likely to be noise
        if w * h < min_area:
            continue
        x_min, y_min, x_max, y_max = add_border(x, y, x+w, y+h, width, height)
        bounding_boxes.append(((x_min, y_min, x_max, y_max), get_orientated_bounding_box(c)))
    return bounding_boxes


def segment_icons(image):
    image_segments = []
    # get bounding boxes for the image
    bounding_boxes = get_bounding_boxes(image.copy())
    # filter out boxes that are entirely covered by a larger bounding box
    bounding_boxes = filter_covered_boxes(bounding_boxes)

    for (x_min, y_min, x_max, y_max), bbx in bounding_boxes:
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
        image_segments.append((padded_image, bbx))
    return image_segments


def sift_detect_and_compute(sift_params, image, resize=None, return_vars=None):
    # calculate keypoints and descriptors for the image (resize if necessary)
    sift = cv2.SIFT_create(**sift_params)
    if resize:
        image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_LINEAR)
    kp, desc = sift.detectAndCompute(image, None)
    return kp, desc, return_vars


def segment_detect_and_compute(sift_params, image, resize, return_vars=None):
    segments = segment_icons(image.copy())
    segments_kp_desc = []
    # calculate keypoins and descriptors for each segment (i.e. individual icon in the image)
    for image, bounding_box in segments:
        kp, desc, _ = sift_detect_and_compute(sift_params, image, resize)
        if desc is not None:
            segments_kp_desc.append(((kp, desc), image, bounding_box))
    return segments_kp_desc, return_vars


def draw_bounding_box(image, bounding_box, text, colour=(0, 255, 0)):
    # draw icons bounding box and predicted name
    cv2.drawContours(image, [bounding_box], 0, colour, 2)
    # Find the highest point (minimum y-coordinate)
    highest_point = min(bounding_box, key=lambda pt: pt[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (highest_point[0], highest_point[1] - 5)
    cv2.putText(image, text, text_position, font, 0.5, colour, 1, cv2.LINE_AA)


def feature_detection(training_data, query_data, params, show_output=True):

    bf = cv2.BFMatcher(**params['BF'])

    # compute keypoints and descriptors for training data
    with ThreadPoolExecutor(max_workers=n_workers) as executor1:
        futures_train = [executor1.submit(
            sift_detect_and_compute,
            params['SIFT'],
            train_image,
            None,
            feature
        ) for train_image, feature in training_data]

    # collect threading futures and filter out any images that have no descriptors
    all_training_data_kp_desc = []
    for future in futures_train:
        train_kp, train_desc, feature = future.result()
        if train_desc is None:
            print(f'No descriptors found for {feature}')
            continue
        all_training_data_kp_desc.append((feature, train_kp, train_desc))

    # segment query images into individual icons and computer keypoints and descriptors
    with ThreadPoolExecutor(max_workers=n_workers) as executor3:
        futures_query = [executor3.submit(
            segment_detect_and_compute,
            params['SIFT'],
            query_image,
            params['resizeQuery'],
            (path, actual_features)
        ) for path, query_image, actual_features in query_data]

    false_positives = []
    false_negatives = []
    # main loop to match query images to training data
    for future in futures_query:
        segments_kp_desc, (path, actual_features) = future.result()
        if len(segments_kp_desc) == 0:
            print(f'No descriptors found for {path}')
            continue

        query_image = cv2.imread(path)

        predicted_features = []
        for (query_kp, query_desc), segment, bounding_box in segments_kp_desc:
            for feature_name, train_kp, train_desc in all_training_data_kp_desc:
                matches = bf.knnMatch(query_desc, train_desc, k=2)

                # filter out matches using Lowe's ratio test
                good_matches = []
                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        if m.distance < params['ratioThreshold'] * n.distance:
                            good_matches.append(m)

                # at least 4 matches are needed for homography
                if len(good_matches) < 4:
                    continue

                # Extract source (query) and destination (train) keypoints coordinates from good matches
                src_pts = np.float32([train_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Find homography matrix between source and destination points using RANSAC
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, **params['RANSAC'])
                matches_mask = mask.ravel().tolist()

                # Check if the match has more than 'inlierScore' inliers
                if sum(matches_mask) > params['inlierScore']:
                    if show_output:
                        draw_bounding_box(query_image, bounding_box, feature_name)

                    predicted_features.append(feature_name)

        predicted_feature_names_set = set(predicted_features)
        actual_feature_names_set = set([f[0] for f in actual_features])

        # calculate false positives and false negatives
        false_neg_diff = actual_feature_names_set.difference(predicted_feature_names_set)
        false_negatives += list(false_neg_diff)

        false_pos_diff = predicted_feature_names_set.difference(actual_feature_names_set)
        false_positives += list(false_pos_diff)

        if show_output:
            cv2.imshow('image', query_image)
            cv2.waitKey(0)

    print('\nSummary of results:')
    print(f'False positives: {Counter(false_positives).most_common()}')
    print(f'False negatives: {Counter(false_negatives).most_common()}')

    if show_output:
        cv2.destroyAllWindows()


def feature_detection_for_graphing(training_data, query_data, params):
    start_time = time.time()
    training_data_names_set = set(f[1] for f in training_data)

    bf = cv2.BFMatcher(**params['BF'])

    # compute keypoints and descriptors for training data
    with ThreadPoolExecutor(max_workers=n_workers) as executor1:
        futures_train = [executor1.submit(
            sift_detect_and_compute,
            params['SIFT'],
            train_image,
            None,
            feature
        ) for train_image, feature in training_data]

    # collect threading futures and filter out any images that have no descriptors
    all_training_data_kp_desc = []
    for future in futures_train:
        train_kp, train_desc, feature = future.result()
        if train_desc is None:
            continue
        all_training_data_kp_desc.append((feature, train_kp, train_desc))

    # segment query images into individual icons and computer keypoints and descriptors
    with ThreadPoolExecutor(max_workers=n_workers) as executor3:
        futures_query = [executor3.submit(
            segment_detect_and_compute,
            params['SIFT'],
            query_image,
            params['resizeQuery'],
            (path, actual_features)
        ) for path, query_image, actual_features in query_data]

    accuracies = []
    true_positives = 0
    false_positives = []
    # main loop to match query images to training data
    for future in futures_query:
        segments_kp_desc, (path, actual_features) = future.result()
        if len(segments_kp_desc) == 0:
            continue

        predicted_features = []
        for (query_kp, query_desc), _, _ in segments_kp_desc:
            for feature_name, train_kp, train_desc in all_training_data_kp_desc:
                matches = bf.knnMatch(query_desc, train_desc, k=2)

                # filter out matches using Lowe's ratio test
                good_matches = []
                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        if m.distance < params['ratioThreshold'] * n.distance:
                            good_matches.append(m)

                # at least 4 matches are needed for homography
                if len(good_matches) < 4:
                    continue

                # Extract source (query) and destination (train) keypoints coordinates from good matches
                src_pts = np.float32([train_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Find homography matrix between source and destination points using RANSAC
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, **params['RANSAC'])
                matches_mask = mask.ravel().tolist()

                # Check if the match has more than 'inlierScore' inliers
                if sum(matches_mask) > params['inlierScore']:
                    predicted_features.append(feature_name)

        predicted_feature_names_set = set(predicted_features)
        actual_feature_names_set = set([f[0] for f in actual_features])

        # calculate accuracy
        correct_predictions = predicted_feature_names_set.intersection(actual_feature_names_set)

        # calculate false positives and false negatives
        false_pos_diff = predicted_feature_names_set.difference(actual_feature_names_set)
        false_positives += list(false_pos_diff)
        true_positives += len(correct_predictions)

        true_negatives = training_data_names_set - actual_feature_names_set - predicted_feature_names_set

        accuracies.append(round(100 * (len(correct_predictions) + len(true_negatives)) / len(training_data_names_set), 1))

    end_time = time.time()
    avg_time_per_image = round((end_time - start_time) / len(query_data), 3)

    if len(accuracies) == 0:
        accuracies = [0]
    return np.mean(accuracies), len(false_positives), true_positives, avg_time_per_image


def feature_detection_hyperopt(bf, kp_desc_query, train_kp, train_desc, feature_name, params):
    for (query_kp, query_desc), _ in kp_desc_query:

        matches = bf.knnMatch(query_desc, train_desc, k=2)

        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < params['ratioThreshold'] * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            continue

        src_pts = np.float32([train_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, **params['RANSAC'])
        matches_mask = mask.ravel().tolist()

        if sum(matches_mask) > params['inlierScore']:
            return feature_name

    return


def read_test_dataset(dir, file_ext):

    image_files = os.listdir(dir + 'images/')
    image_files = sorted(image_files, key=lambda x: int(x.split("_")[2].split(".")[0]))

    all_data = []
    for image_file in image_files:
        csv_file = dir + 'annotations/' + image_file[:-4] + file_ext
        with open(csv_file, 'r') as fr:
            features = fr.read().splitlines()
        all_features = []
        for feature in features:
            end_of_class_name_index = feature.find(", ")
            end_of_first_tuple_index = feature.find("), (") + 1
            feature_class = feature[:end_of_class_name_index]
            feature_coord1 = eval(feature[end_of_class_name_index + 2:end_of_first_tuple_index])
            feature_coord2 = eval(feature[end_of_first_tuple_index + 2:])

            all_features.append([feature_class, feature_coord1, feature_coord2])
        path = dir + 'images/' + image_file

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        all_data.append((
            path,
            img,
            all_features
        ))

    return all_data


def read_training_dataset(dir):
    training_data = []
    for path in os.listdir(dir):

        if not path.endswith('.png'):
            continue

        img = cv2.imread(dir + path, cv2.IMREAD_GRAYSCALE)
        feature_name = feature_name_from_path(path)
        training_data.append((img, feature_name))

    return training_data


def remove_noise_from_image(image, kernel=np.ones((3, 3), np.uint8)):
    _, thresh_img = cv2.threshold(image, 250, 255, cv2.THRESH_TOZERO_INV)
    eroded_img = cv2.erode(thresh_img, kernel, cv2.BORDER_REFLECT)
    mask = np.uint8(eroded_img <= 20) * 255
    result = cv2.bitwise_or(eroded_img, mask)
    return result


def feature_name_from_path(img_path):
    return img_path[img_path.find('-')+1:img_path.find('.png')]
