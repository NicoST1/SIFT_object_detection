import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
from basic_sift import feature_detection_hyperopt, segment_detect_and_compute, read_test_dataset, read_training_dataset, sift_detect_and_compute


RED = '\u001b[31m'
GREEN = '\u001b[32m'
NORMAL = '\u001b[0m'

task3_no_rotation_images_dir = 'Task2Dataset/TestWithoutRotations/'
task3_rotated_images_dir = 'Task3Dataset/'
task3_training_data_dir = 'Task2Dataset/Training/'

try:
    all_no_rotation_images_and_features = read_test_dataset(task3_no_rotation_images_dir, '.txt')
    all_rotation_images_and_features = read_test_dataset(task3_rotated_images_dir, '.csv')
    all_training_images_and_paths = read_training_dataset(task3_training_data_dir)
    print()
except Exception as e:
    print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
    exit()


def objective_false_res(param):
    bf = cv2.BFMatcher(crossCheck=False, normType=2)

    with ThreadPoolExecutor(max_workers=8) as executor1:
        futures_train = [executor1.submit(sift_detect_and_compute, param['SIFT'], train_image, None, feature) \
                         for train_image, feature in all_training_images_and_paths]

    all_training_data_kp_desc = []
    for future in futures_train:
        train_kp, train_desc, feature = future.result()
        if train_desc is None:
            return {'status': STATUS_FAIL}
        all_training_data_kp_desc.append((feature, train_kp, train_desc))

    total_results = 0
    false_results = 0
    false_positives = []
    false_negatives = []
    wrong_images = []

    with ThreadPoolExecutor(max_workers=8) as executor2:
        futures_query = [executor2.submit(segment_detect_and_compute, param['SIFT'], query_image, param['resizeQuery'], (path, actual_features)) \
                         for path, query_image, actual_features in all_rotation_images_and_features]

    for future in futures_query:
        kp_desc_query, (path, actual_features) = future.result()

        if len(kp_desc_query) == 0:
            return {'status': STATUS_FAIL}

        with ThreadPoolExecutor(max_workers=8) as executor3:
            futures_match = [executor3.submit(feature_detection_hyperopt, bf, kp_desc_query, train_kp, train_desc, feature_name, \
                                              param) for feature_name, train_kp, train_desc in all_training_data_kp_desc]

        predicted_features = [f.result() for f in futures_match if f.result() is not None]
        predicted_feature_names_set = set(predicted_features)
        actual_feature_names_set = set([f[0] for f in actual_features])

        total_results += len(actual_feature_names_set)
        if predicted_feature_names_set == actual_feature_names_set:
            continue

        wrong_images.append(path)

        false_pos_diff = predicted_feature_names_set.difference(actual_feature_names_set)
        false_neg_diff = actual_feature_names_set.difference(predicted_feature_names_set)

        false_positives += list(false_pos_diff)
        false_negatives += list(false_neg_diff)

        false_results += len(false_pos_diff) + len(false_neg_diff)

    accuracy = (total_results-false_results) / total_results
    if false_results < 10:
        print(f'Current Accuracy: {accuracy}, loss: {false_results}')
        print('Current Params: ' + str(param))
        print('false positives', Counter(false_positives))
        print('false negatives', Counter(false_negatives))
        print('wrong images', wrong_images)
    return {'loss': false_results, 'status': STATUS_OK, 'model': param}

try:

    param_space = {
        'RANSAC': {
            'ransacReprojThreshold': hp.uniform('ransacReprojThreshold', 10.5, 12.5),
        },
        'ratioThreshold': hp.uniform('ratioThreshold', 0.55, 0.75),
        'resizeQuery': hp.choice('resizeQuery', range(90, 100)),
        'inlierScore': hp.choice('inlierScore', range(3, 6)),
        'SIFT': {
            'nOctaveLayers': hp.choice('nOctaveLayers', range(5, 8)),
            'contrastThreshold': hp.uniform('contrastThreshold', 0.003, 0.005),
            'edgeThreshold': hp.uniform('edgeThreshold', 15.5, 17.5),
            'nfeatures': hp.choice('nfeatures', range(1600, 1901, 100)),
            'sigma': hp.uniform('sigma', 2, 2.5),
        },
    }

    test_dataset = all_no_rotation_images_and_features + all_rotation_images_and_features
    trials = Trials()
    try:
        fmin(
            fn=objective_false_res,
            space=param_space,
            algo=tpe.suggest,
            max_evals=500,
            trials=trials
        )
    except:
        pass

    best_params = trials.best_trial['result']['model']
    print('Best parameters:', best_params)
except Exception as e:
    print(RED, 'Unknown error occurred:', NORMAL, traceback.format_exc())
    exit()
