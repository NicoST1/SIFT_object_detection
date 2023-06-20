from collections import Counter
import os
import cv2
import re
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from segment_img import segment_icons
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe


class ImageDataBase:

    def __init__(self, sift_params, n_clusters, training_directory):
        self.sift_params = sift_params
        self.n_clusters = n_clusters

        self.documents = self.initialise_documents(training_directory)

        flatten_descriptors = [(key, v) for key, value in self.documents['descriptors'].items() for v in value]

        self.kmeans = self.initialise_clustering(flatten_descriptors)
        self.document_term_freq = self.initialise_document_term_freq(flatten_descriptors)

        self.inv_term_freq = self.initialise_inv_term_freq()

        self.df_idf = self.document_term_freq.apply(lambda row: row / sum(row) \
                                                                * np.log(
            len(self.document_term_freq) / self.inv_term_freq['frequency'].T), axis=1)


    def initialise_documents(self, directory):
        sift = cv2.SIFT_create(**self.sift_params)
        documents = pd.DataFrame(columns=['keypoints', 'descriptors'])

        for filename in os.listdir(directory):
            if filename.endswith('.png'):
                img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
                match = re.search(r'\d+-([\w-]*)\.png', filename)
                if match:
                    name = match.group(1)
                    kp, desc = sift.detectAndCompute(img, None)
                    documents.loc[name] = [kp, desc]
        return documents

    def initialise_clustering(self, flatten_descriptors):
        kmeans = KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=42)
        kmeans.fit(list(zip(*flatten_descriptors))[1])
        return kmeans

    def initialise_document_term_freq(self, flatten_descriptors):
        document_term_freq = pd.DataFrame(columns=range(self.n_clusters))

        for i in range(len(flatten_descriptors)):
            if flatten_descriptors[i][0] not in document_term_freq.index:
                document_term_freq.loc[flatten_descriptors[i][0]] = [0] * self.n_clusters
            document_term_freq.loc[flatten_descriptors[i][0]][self.kmeans.labels_[i]] += 1
        return document_term_freq

    def initialise_inv_term_freq(self):
        inv_term_freq = pd.DataFrame(columns=['frequency', 'document_ids'])

        for i in range(self.n_clusters):
            documents = self.document_term_freq[self.document_term_freq[i] > 0].index.tolist()
            freq = len(documents)

            inv_term_freq.loc[i] = [freq, documents]

        return inv_term_freq


def draw_bounding_box(image, bounding_box, text, colour=(0, 255, 0)):
    # draw icons bounding box and predicted name
    cv2.drawContours(image, [bounding_box], 0, colour, 2)
    # Find the highest point (minimum y-coordinate)
    highest_point = min(bounding_box, key=lambda pt: pt[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (highest_point[0], highest_point[1] - 5)
    cv2.putText(image, text, text_position, font, 0.5, colour, 1, cv2.LINE_AA)


class ImageQuery:

    def read_query_images(self, query_directory):
        image_files = os.listdir(query_directory + 'images/')
        image_files = sorted(image_files, key=lambda x: int(x.split("_")[2].split(".")[0]))

        all_data = []
        for image_file in image_files:
            csv_file = os.path.join(query_directory + 'annotations/', image_file[:-4] + '.csv')

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

            path = os.path.join(query_directory + 'images/', image_file)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            img_segments = segment_icons(img)

            all_data.append((
                path,
                img_segments,
                all_features
            ))

        return all_data

    def get_feature_vector(self, query_desc):
        feature_vector = pd.Series([0] * self.img_database.n_clusters)

        for desc in query_desc:
            min_dist = np.inf
            min_label = 0

            for i, centroid in enumerate(self.img_database.kmeans.cluster_centers_):
                dist = np.linalg.norm(desc - centroid)
                if dist < min_dist:
                    min_dist = dist
                    min_label = i

            feature_vector[min_label] += 1
        feature_vector = (feature_vector / sum(feature_vector)) * np.log(len(self.img_database.document_term_freq) \
                                                                         / self.img_database.inv_term_freq[
                                                                             'frequency'].T)
        return feature_vector

    def match_image(self, img_segments, features, ransac_params, resize_query, path):

        sift = cv2.SIFT_create(**self.img_database.sift_params)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        predictions = []
        for segment, bbox in img_segments:
            segment = cv2.resize(segment, (resize_query, resize_query), interpolation=cv2.INTER_LINEAR)
            query_kp, query_desc = sift.detectAndCompute(segment, None)
            feature_table = self.img_database.df_idf.copy(deep=True)
            feature_vector = self.get_feature_vector(query_desc)

            feature_table['cosine'] = cosine_similarity(feature_table, [feature_vector])

            max_candidate = None
            max_score = 0
            for candidate in feature_table.sort_values(by='cosine', ascending=False).head(10).index:

                train_kp, train_desc = self.img_database.documents.loc[candidate, ['keypoints', 'descriptors']]

                matches = bf.match(query_desc, train_desc)

                if len(matches) < 4:
                    continue

                src_pts = np.float32([train_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([query_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, **ransac_params)
                score = sum(mask)

                if score > max_score:
                    max_score = score
                    max_candidate = candidate

            predictions.append((max_candidate, bbox, path))

        return predictions, features

    def match_images(self, param_space, show_output=True):

        self.query_images = self.read_query_images(param_space['query_directory'])
        self.img_database = ImageDataBase(param_space['SIFT'], param_space['n_clusters'], param_space['training_directory'])
        self.top_k_percent = 0.1

        false_positives = []
        false_negatives = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.match_image, img_segments, features, param_space['RANSAC'],
                                       param_space['resize_query'], path) \
                       for path, img_segments, features in self.query_images]

        for future in futures:
            predictions, features = future.result()

            feature_set = set([f[0] for f in features])
            prediction_set = set([p[0] for p in predictions])

            false_neg_diff = feature_set.difference(prediction_set)
            false_negatives += list(false_neg_diff)

            false_pos_diff = prediction_set.difference(feature_set)
            false_positives += list(false_pos_diff)

            if show_output:

                query_image = cv2.imread(predictions[0][-1], cv2.IMREAD_COLOR)

                for feature_name, bbox, _ in predictions:
                    draw_bounding_box(query_image, bbox, feature_name)

                cv2.imshow('image', query_image)
                cv2.waitKey(0)


        if show_output:
            cv2.destroyAllWindows()


        print('\nSummary of results:')
        print(f'False positives: {Counter(false_positives).most_common()}')
        print(f'False negatives: {Counter(false_negatives).most_common()}')

        return len(false_negatives), len(false_positives)


if __name__ == '__main__':

    img_query = ImageQuery()

    param_space = {'RANSAC': {'ransacReprojThreshold': 10.520573120773145},
                   'SIFT': {'contrastThreshold': 0.003184605067194348,
                            'edgeThreshold': 19.39167977800373,
                            'nOctaveLayers': 7,
                            'nfeatures': 1800,
                            'sigma': 4.045882620339618},
                   'n_clusters': 430,
                   'query_directory': 'Test/',
                   'resize_query': 180,
                   'training_directory': 'Training/'
                   }

    param_space = {
        'training_directory': 'Training/',
        'query_directory': 'Test/',
        'resize_query': hp.choice('resizeQuery', range(80, 160, 1)),
        'n_clusters': hp.choice('n_clusters', range(400, 460, 1)),
        'RANSAC': {
            'ransacReprojThreshold': hp.uniform('ransacReprojThreshold', 8, 12),
        },
        'SIFT': {'nOctaveLayers': hp.choice('nOctaveLayers', range(5, 8)),
                 'contrastThreshold': hp.uniform('contrastThreshold', 0.002, 0.004),
                 'edgeThreshold': hp.uniform('edgeThreshold', 17, 22),
                 'nfeatures': hp.choice('nfeatures', range(1700, 2100, 100)),
                 'sigma': hp.uniform('sigma', 1.5, 5),
                 },
    }

    trials = Trials()
    try:
        fmin(
            fn=img_query.match_images,
            space=param_space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials
        )
    except:
        pass

    best_params = trials.best_trial['result']['model']

    print(best_params)


