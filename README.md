# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Vehicle Detection Project**

## Udacity Self Driving Car Engineer Nanodegree Project:

### Goal

The goal of this project is to design and implement software pipeline to detect vehicles in a video. As a result, this Jupyter notebook should be able to playback an .mp4 file of a car driving down the highway, identifying all vehicles in its view with bounding boxes.

### Introduction
### Vehicle Detection Pipeline

1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
2. Train a Linear Support Vector Machine (SVM) Classifier.
3. Search for vehicles in a given .mp4 video frame using a sliding-window technique on the Linear SVM Classifier.
4. Because the sliding-window technique makes duplicate detections of vehicles (changing window sizes to identify small to large vehicles in a given video frame), create a heat map of detected vehicles, rejecting those beyond a certain threshold.
5. Draw bounding boxes around detected vehicles from the heat map and display it on the given video frame.
6. Finally, playback the video to see the detected vehicles!

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./examples/RandomCarNotCar.jpg
[image9]: ./examples/CrudeDetector.jpg
[image10]: ./examples/HeatMap.jpg
[image11]: ./examples/RefinedDetector.jpg
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.


The code for this step is contained in the second code block of the Jupyter notebook under the # TEST MATPLOTLIB TO DISPLAY CAR AND NOT CAR IMAGES comment.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Below the random images is an example of the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)` representing random car and not car images:

![alt text][image8]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that the following parameters aided in a 98%+ accuracy in my Linear SVC. 

43.06 Seconds to extract HOG features
Orientations: 11 (Tested between 5 and 15)
Pixels Per Cell: 16 (Teseted between (8 and 64 in multiples of 8)
Cells Per Block: 2 (Tested between 2 and 8 in multioples of 2)
Feature Length: 1188
1.04 Seconds to train SVC
Test Accuracy of SVC =  0.9803 (98%+ accuracy)
10 Linear SVC predictions:  [ 0.  0.  0.  1.  1.  0.  1.  0.  0.  1.]
10 Test Labels:             [ 0.  0.  0.  1.  1.  0.  1.  0.  0.  1.]
0.00447 Seconds to predict 10 labels with SVC

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I created 2D features and labels vectors whose lengths are simply the number of car features and not_car features, respectively.

X = np.vstack((car_features, notcar_features)).astype(np.float64)  
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

I initilized a Linear SVC

svc = LinearSVC()

Then I randomly shuffled the feature sets in order to train them on the Linear SVC

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)
svc.fit(X_train, y_train)

The above code is located in the middle of the 2nd cell.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions with scales ranging between 1.0 and 3.0 in increments of 0.5:

### ystarts    = [400,416,400,432,400,432,400,464]
### ystops     = [464,480,496,528,528,560,596,660]
### scales     = [1.0,1.0,1.5,1.5,2.0,2.0,3.5,3.5]

The above positions and scales are located under the #TEST DRAW BOXES# comment in the 3rd cell.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales using YUV 3-channel ("ALL") HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image9]
![alt text][image10]
![alt text][image11]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. Code to implmement this is located in the 3rd cell of the Jupyter notebook:


labels = label(heatmap_img)
plt.title("Heat Mapped Boxes")
plt.figure(figsize=(10,10))
plt.imshow(labels[0], cmap='gray')
print('Found', labels[1], 'cars.')

draw_img, rect = draw_labeled_bboxes(np.copy(test_img), labels)
plt.title("Grayed Threshold Heat Mapped Boxes")
plt.figure(figsize=(10,10))
plt.imshow(draw_img)
plt.title("Refined Identified Vehicles")


Here's an example result showing the heatmap from a series of frames of video, the result of `label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

Here is the code to run the proc_frame function over a series of frames in the input video:

test_out_file = 'project_video_out.mp4'
clip_test = VideoFileClip('project_video.mp4')
clip_test_out = clip_test.fl_image(process_frame)
%time clip_test_out.write_videofile(test_out_file, audio=False)
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

With the help of Udacity's Self Driving Car Nanodegree Engineer program modular instructions (emcompassed in 41 sub lessons), I was able to configure a utility.py file with functions that aid in my Vehicle Detector Pipeline. Such functionally enabled me to do the following:
Set up a Linear Support Vector Machine classifier
Extract HOG features from multiple image sets
Find cars by returning sets of rectangles in an image where a car might be expected
Draw boxes of various sizes on the respective image
Remove duplicate detections using a heat map with a threshold
Redraw boxes around said heat mapped images
Display the image with well-defined boxes around correctly detected vehicles
Iterate over a stream of images in a video to detect vehicles in real time!
Some drawbacks to my solution include the fact that there were some test images where a section of an image with an object was incorrectly labeled as a car. The detector mistook a green bush as an incoming car on the left side of the highway. No matter how I changed my classifier's parameters, the green bush was still identified to be a car. I'll use a different classifier (like a modified decision tree or a different support vector classifier) if I were to refine this project very soon. Until then, I had a blast figuring out how to use the sk framework to classify and detect vehicles.

### References

1. https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
2. http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
3. http://scikit-learn.org/stable/modules/svm.html
4. http://scikit-learn.org/stable/model_selection.html#model-selection
5. http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
6. http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing

