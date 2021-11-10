# human-detection


## Install
pipenv install

## Run
pipenv run python detect.py --image data/video/dior.mp4 --skip 20

## Code
The output are then stored in data/ouput/dior.mp4

Our approach is to detect people and then track them. The user can choose the detection frequency with the --skip_frames parameter at runtime.
L'algorith
The algorithm takes about 4 minutes to run the DIOR promotional video.

## Areas for Improvement

#### 1. Accelerate the algorithm
##### 1.1 Reduce the size of the image before detection
##### 1.2 Define regions of interest to make the algorithm focus on specific areas of the text
##### 1.3 Compress the image

2. Detection
2.1 Testing better pre-trained models
2.2 Image preprocessing to isolate the foreground
2.3 Redefining the confidence threshold for classification 

3. Tracking
We could implement a more interesting version to identify a person when he reappears in totally different contexts (location, glasses, clothes, position). This could materialize with the implementation of facial recognition when possible.
