import os
import traceback

import cv2
import numpy as np
from basic_sift import feature_detection_for_graphing, read_test_dataset, read_training_dataset, remove_noise_from_image
from matplotlib import pyplot as plt


# Initial setup --------------------------------------------------------------------------
dir = 'Task3/report_assets'
if not os.path.exists(dir):
    os.makedirs(dir)

train_data_dir = 'Task3/Task2Dataset/Training/'
query_data_dirs = [
    ('Task3/Task3Dataset/', '.csv'),
    ('Task3/Task2Dataset/TestWithoutRotations/', '.txt'),
]

try:
    train_data = read_training_dataset(train_data_dir)
    query_data = []
    for q_dir, ext in query_data_dirs:
        for data in read_test_dataset(q_dir, file_ext=ext):
            query_data.append(data)
except Exception as e:
    print('Error while reading datasets:', traceback.format_exc())
    exit()

best_params = {
    'RANSAC': {
        'ransacReprojThreshold': 11.110294305510669
    },
    'SIFT': {
        'contrastThreshold': 0.0039052330228148877,
        'edgeThreshold': 16.379139206562137,
        'nOctaveLayers': 6,
        'nfeatures': 1700,
        'sigma': 2.2201211013686857
    },
    'BF': {
        'crossCheck': False,
        'normType': 2
    },
    'inlierScore': 4,
    'ratioThreshold': 0.6514343913409797,
    'resizeQuery': 95
}

sift = cv2.SIFT_create(**best_params['SIFT'])
bf = cv2.BFMatcher(**best_params['BF'])


# Get noise vs no noise image -----------------------------------------------------------
print('Creating noise removal example...')
gray_example_image = query_data[0][1]
no_noise_example_image = remove_noise_from_image(gray_example_image)
vis = np.concatenate(
    (
        cv2.copyMakeBorder(
            gray_example_image,
            top=4,
            bottom=4,
            left=4,
            right=2,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        ),
        cv2.copyMakeBorder(
            no_noise_example_image,
            top=4,
            bottom=4,
            left=4,
            right=2,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        ),
    ),
    axis=1
)
cv2.imwrite(f'{dir}/noise_removal_example.png', vis)


# Determine "Accuracy vs Parameter", "Mean Average Error vs Parameter & Average time per image vs Parameter"  -----------------------------------------------------------
def create_subplots(field, var, accuracies, fps_and_tps, times):
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    plt.subplots_adjust(wspace=0.6)

    # Plot accuracy
    axs[0].plot(var, accuracies, label='Accuracy')
    axs[0].set_xlabel(field)
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title(f'Accuracy vs. {field}')

    # Plot False Positives and True Positives
    false_pos = [i[0] for i in fps_and_tps]
    true_pos = [i[1] for i in fps_and_tps]
    axs[1].plot(var, false_pos, color='blue')
    axs[1].set_xlabel(field)
    axs[1].set_ylabel('False Positives', color='blue')
    axs[1] = axs[1].twinx()
    axs[1].plot(var, true_pos, color='yellow')
    axs[1].set_ylabel('True Positives', color='yellow')
    axs[1].set_title(f'False/True Positives vs. {field}')

    # Plot Time
    axs[2].plot(var, times, label='Time')
    axs[2].set_xlabel(field)
    axs[2].set_ylabel('Time taken per image')
    axs[2].set_title(f'Time taken per image vs. {field}')

    return fig


def get_sensitivity(field_name, param_space, values):
    accuracies = []
    fps_and_tps = []
    times = []

    for param in param_space:
        params = best_params.copy()
        for key, val in param.items():
            if key in params:
                if isinstance(val, dict):
                    params[key] = {**params[key], **val}
                else:
                    params[key] = val

        accuracy, false_positives, true_positives, avg_time_per_image = feature_detection_for_graphing(
            train_data,
            query_data,
            params
        )

        if np.isnan(accuracy):
            accuracy = 0

        accuracies.append(accuracy)
        fps_and_tps.append((false_positives, true_positives))
        times.append(avg_time_per_image)

    return create_subplots(field_name, values, accuracies, fps_and_tps, times)

param_spaces_to_try = [
    ('ransacReprojThreshold', [{'RANSAC': { 'ransacReprojThreshold': x }} for x in np.arange(0, 30)], np.arange(0, 30)),

    ('contrastThreshold', [{'SIFT': { 'contrastThreshold': x }} for x in np.arange(0, 2, 0.1)], np.arange(0, 2, 0.1)),
    ('edgeThreshold', [{'SIFT': { 'edgeThreshold': x }} for x in np.arange(1, 30)], np.arange(1, 30)),
    ('nOctaveLayers', [{ 'SIFT': { 'nOctaveLayers': x }} for x in range(1, 10)], list(range(1, 10))),
    ('nfeatures', [{'SIFT': { 'nfeatures': x }} for x in range(0, 3000, 100)], list(range(0, 3000, 100))),
    ('sigma', [{'SIFT': { 'sigma': x }} for x in range(1, 11)], list(range(1, 11))),

    ('inlierScore', [{'inlierScore': x} for x in range(0, 10)], list(range(0, 10))),
    ('ratioThreshold', [{'ratioThreshold': x} for x in np.arange(0.1, 1.5, 0.1)], np.arange(0.1, 1.5, 0.1)),
    ('resizeQuery', [{'resizeQuery': x} for x in range(50, 101)], list(range(50, 101))),
]

for param_name, param_space, values in param_spaces_to_try:
    print(f'Evaluating parameter: {param_name}...')
    fig = get_sensitivity(param_name, param_space, values)
    plt.savefig(f'{dir}/{param_name}_plot.png')
    plt.close(fig)


print('Done.')
