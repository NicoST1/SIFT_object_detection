import traceback
from basic_sift import feature_detection, read_test_dataset, read_training_dataset
from BoVW_sift import ImageQuery
import sys


RED = '\u001b[31m'
NORMAL = '\u001b[0m'

if __name__ == '__main__':

    arg = 'basic'

    params_basic = {
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

    params_bovw = {'RANSAC': {'ransacReprojThreshold': 10.520573120773145},
                   'SIFT': {'contrastThreshold': 0.003184605067194348,
                            'edgeThreshold': 19.39167977800373,
                            'nOctaveLayers': 7,
                            'nfeatures': 1800,
                            'sigma': 4.045882620339618},
                   'n_clusters': 430,
                   'query_directory': 'Test/',
                   'resize_query': 180,
                   'training_directory': 'Train/'
                   }

    if len(sys.argv) > 1:
        argument = sys.argv[1]

        if argument == 'basic':
            print('Running basic SIFT... (This may take a while)')

            train_data_dir = 'Train/'
            query_data_dirs = [
                ('Test/', '.csv'),
            ]

            try:
                train_data = read_training_dataset(train_data_dir)
                query_data = []
                for q_dir, ext in query_data_dirs:
                    for data in read_test_dataset(q_dir, file_ext=ext):
                        query_data.append(data)
            except Exception as e:
                print(RED, 'Error while reading datasets:', NORMAL, traceback.format_exc())
                exit()

            print('* Press Enter to continue')
            feature_detection(train_data, query_data, params_basic, show_output=True)

        elif argument == 'bovw':
            print('Running BoVW SIFT... (This may take a while)')

            img_query = ImageQuery()
            try:
                print('* Press Enter to continue')
                img_query.match_images(params_bovw, show_output=True)
            except KeyboardInterrupt:
                print('Interrupted by user')
                exit()

        else:
            print('Invalid argument')
            exit()




