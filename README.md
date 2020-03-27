# Object Detection And Classification For Species Recognition
by Jamie McElhiney
## Problem Statement

Around 12% of current bird species are considered endangered, threatened or vulnerable in some way. Many species are referred to as 'keystone species' which indicates that they hold an invaluable presence in their respective ecosystem. 
Many birds aid in pollination, create subhabitats such as nests or entry points in trees, and are host to a variety of parasites and flies that also hold weight within their ecosystem. It is important to conserve these species for both ecological and humane reasons.

Habitat loss and destruction are one of the leading causes of bird species extinction. We can discretely place cameras in areas we know there may be endangered species and are being considered for deforestation or other means of habitat disruption. We will attempt to train our own bird detection system from scratch so that we can appropriately assess the population and activity of specific species within our geographical regions of study. To do this, we will segment out our bird images with a selective search algorithm and use a CNN to classify these segments as foreground(our bird) or background. We will evaluate our model on binary accuracy.

## Executive Summary

Order of notebooks:  
SpeciesRecognitionDataCollection_0 (Creating dataframe and writing images-already done)  
SpeciesRecognitionSelectiveSearch_1 (Object detection model using selective search)  
SpeciesRecognitionNMS_2 (Non-max suppression and species classification)  

We used a combination of images from Cal-Tech's bird data set . Cal-Tech's bird data set includes 200 species of around 80 images per species. We have abridged the dataset to 6 species of around 60 images per species. The species we will be considering include a Cardinal, Gadwall, Horned Grebe, Blue Jay, Red-winged Blackbird, and Western Meadowlark. The intended outcome of our study is to properly implement an object detection model using selective search, followed by a classification model. Selective search is an image segmentation algorithm which allows us to extract and separate different regions within our image. 

Selective Search
- Diverse application of grouping/segmentation algorithms based on color/texture/shape  
- Proposes multiple regions of interest  
- Heirarchical 

As with many image recognition and object detection problems, we run into many recurrent and expected problems. We must consider different perspective, orientation of object, brightness, noise, and so on. If we train our model on image of birds only facing one way, we will have a harder time predicting and classifying birds facing any other direction. We also have to considered different background classes. . We have accounted for some of these issues with data augmentation where we have generated copies of our original data with added noise that can be considered synonymous with camera inconsistencies (scratching,dust) and orientation flips (pose left,right). Our augmentation was minimal given time constraints and annotations, so we are avoiding rotation.

For both approaches we fit and evaluated a classification model in a binary setting (foreground,background). We use our annotated boundary boxes on our birds so we can retrieve suggested candidate regions for foreground objects. We calculated an Intersection over Union score (IoU) to determine how overlapped our proposed regions by our segmentation are with our ground truth (annoted labels) box. Regions with a score greater than 0.7 would be classified as foreground and regions with a score less than 0.3 would be classified as background. Scores between 0.7 and 0.3 are ignored. Our classification model consists out a multi layer Convalutional Network connected to a Fully Connected network.


## Data Dictionary


| Name| Data Types (Pandas) | Description |
|---|---|---|
|root|object|root file path of image|
|filename|object|full filename plus doc ext of image file|
|w|int64|width of image|
|h|int64|height of image|
|x1|int64|first x-coordinate of bounding box|
|y1|int64|first y-coordinate of bounding box|
|x2|int64|second x-coordinate of bounding box|
|y2|int64|second y-coordinate of bounding box|

## Results
 
<img src=./images/Capture30.jpg width="220" height="200"><img src=./images/Capture24.jpg width="220" height="200">Proper localization, 1 incorrect classification  
Detection--> NMS --> Classification

<img src=./images/Capture32.jpg width="160" height="200"><img src=./images/Capture33.jpg width="160" height="200">Correct prediction    
Detection--> NMS --> Classification

 
<img src=./images/Capture35.jpg width="300" height="200">Semi-correct localization, correct classification.

<img src=./images/Capture29.jpg width="280" height="240">  
Correct prediction


## Conclusion and Recommendations.

We trained a binary classification CNN for object detection achieving 95% testing accuracy and 98.8% training accuracy, which is significantly higher than our 72.6% threshold. Looking over our false positives and false negatives, our model still needs some fine tuning as the presence of false positives will greatly interfere with our detection process. A relatively small sample size was used, we had around 300 images that segmented out to around 22k observations with a 17k/5k train test split.

We then trained a multi a multi-classification CNN for species classification achieving a training accuracy of 90% and testing accuracy of 86.8%. This is also siginificantly higher than our 16.67% accuracy threshold.

We noticed our positive samples are not perfect pixel-for-pixel representations of what we are actually detection, our bird. Our samples are bounding boxes that also bring in random noise. So different species of birds will tend to have different background noise that we will also be classifying on.

We made the realization that the presence of false positives is detrimental to detection. We can use our AUC-ROC curve to reevaluate our probability threshold for predictions and adjust True Positive and False Positive tradeoff for stronger deployment. A higher threshold (>0.9) reduced false positive detection and still provided enough true positives to work with.

For further steps and more time, we can obtain more data. The biggest limitation to this project was the time constraint which prevented me from colleting and labeling enough data for an object detector and species classifier. Typically, tens of thousands of images should be used, whereas we only have 200 training images. We can also gridsearch parameters, and use ModelCheckpoints to tune our CNN and deploy it in the setting we are targeting. 

## Sources
- http://www.huppelen.nl/publications/selectiveSearchDraft.pdf
- https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
- https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55
- https://www.pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/
- https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
