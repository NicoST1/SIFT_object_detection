# Object Detection using SIFT

This project uses the Scale-Invariant Feature Transform (SIFT) for detecting and recognizing labeled icons in images. Two approaches are implemented:

1. **Basic Approach:** This approach involves segmenting the images and using SIFT to identify keypoints.
2. **Bag of Visual Words (BoVW) Approach:** This approach identifies objects in images based on keypoint correspondence and maintains spatial consistency using a homography via RANSAC.

## Bag of Visual Words Approach
In the BoVW approach, the keypoint descriptors in the training images are clustered, and the centroids are used to represent the "visual words" in the vocabulary. The keypoints in the query images are normalized to their corresponding centroid and similarity between images is measured using cosine similarity of the corresponding TF-IDF vectors.

To ensure spatial consistency among the matches, RANSAC and homography are applied to the top matches to find the corresponding object in the query image.

## Getting Started

### Installing Necessary Packages

Install necessary packages:
```
pip install -r requirements.txt
```
Python 3.10 was used.

### Using Object Detection using SIFT

To use Object Detection using SIFT, follow these steps:

To run the basic version:
```
python main.py basic
```

To run the BoVW version:
```
python main.py bovw
```

After running either of the commands above, you will see a collection of images with the bounding box and label. To move to the next image, press any key.

