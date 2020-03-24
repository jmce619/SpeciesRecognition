# Object Detection And Classification For Species Recognition
by Jamie McElhiney
## Problem Statement

Around 12% of current bird species are considered endangered, threatened or vulnerable in some way. Many birds play a crucial roll in ecological systems for both predator and prey. Bird species such as these are known as 'keystone species' that hold an invaluable presence in their respective ecosystems. For example, a lot of birds are host to a variety of parasites and flys that also hold weight within their food chains. Additionally, many birds aid in pollination of certain plant species and creation of subhabitats such as nests or entry points in trees (woodpecker). It is incredibly important to conserve these species for both ecological and humane reasons. Not to mention, birds are peculiar and amusing to us. It would be a shame to go out on a walk and not hear the specific chirping melody based on the regional species.  
What we can do is propose a habitat protection movement. Habitat loss and destruction are one of the leading causes of bird species extinction. We can discretely place cameras in areas we know there may be endangered species and are being considered for deforestation or other means of habitat disruption. We will attempt to train our own bird detection system so that we can appropriately assess the population and activity of specific species within our geographical regions of study. Working with local governments and species conservation groups, we can present our data and work against habitat loss.

## Executive Summary

Order of notebooks:
SpeciesRecognitionDataCollection_0 (Creating dataframe and writing images-already done)
SpeciesRecognitionSelectiveSearch_1 (Object detection model using selective search)
SpeciesRecognitionNMS_2 (Non-max suppression and species classification)

We used a combination of images from Cal-Tech's bird data set . Cal-Tech's bird data set includes 200 species of around 80 images per species. We have abridged the dataset to 6 species of around 40 images per species due to time constraints and labeling. The species we will be considering include a Cardinal, Blue Jay, Gadwall,Horned_grebbe, red-winged blackbird and Western Meadowlark. The intended outcome of our study is to deploy a highly accurate object detection system that allows us to localize our bird on an image, as opposed to produced a highly accurate bird classification model (although we will include classification). Image segmentation allows us to extract and separate different regions within our image. We are going to apply a selective search approach. We will then apply and evaluate a Convolutional Neural Network that will help us determine which segments are of interest (bird) and which are not (background noise, other animals,..so on)

Images were labeled in CVAT (Computer Vision Annotation tool) and LabelImg. Augmentations were performed in Roboflow.ai.

Selective Search
- Diverse application of grouping/segmentation algorithms based on color/texture/shape  
- Proposes multiple regions of interest  
- Heirarchical 

As with many image recognition and object detection problems, we run into many recurrent and expected problems. We must consider how we will account for birds with different poses, facing different directions, leaning forwards or backwards, facing the camera or facing away, flying or standing, and so on. If we feed in images of birds in only certain poses, lets say all facing left, how will our detection system recognize a bird facing right. We also have to considered different background classes. Different bird species will habit different genres of terrain(what region are these species found in), and similarly we have to account for what season it is, weather conditions, time of day, lighting, etc.. We have accounted for some of these issues with data augmentation where we have generated copies of our original data with added noise that can be considered synonymous with camera inconsistencies (scratching,dust) and orientation flips (pose left,right). Our augmentation was minimal given time constraints and annotations, so we are avoiding rotation.

For both approaches, we fit and evaluated various classification algorithms in a binary setting (foreground,background). We use our annotated boundary boxes on our birds so we can retrieve suggested candidate regions for foreground objects. We calculated an Intersection over Union score (IoU) to determine how overlapped our proposed regions by our segmentation are with our ground truth (annoted labels) box. Regions with a score greater than 0.7 would be classified as foreground and regions with a score less than 0.3 would be classified as background. Scores between 0.7 and 0.3 are ignored. Our classification model consists out a multi layer Convalutional Network connected to a Fully Connected network.


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
[Detection](./images/Capture30.png)   
[Classification and Non-Max Suppression](./images/Capture24.png)  

[Detection](./images/Capture32.png) 
[Classification and Non-Max Suppression](./images/Capture33.png) 

[Semi-correct detection](./images/Capture35.png)   

[Correct detection and classification](./images/Capture29.png) 


## Conclusion and Recommendations.

We trained a binary classification CNN for object detection achieving 95% testing accuracy and 98.8% training accuracy, which is significantly higher than our 72.6% threshold. Looking over our false positives and false negatives, our model still needs some fine tuning as the presence of false positives will greatly interfere with our detection process. A relatively small sample size was used, we had around 300 images that segmented out to around 22k observations with a 17k/5k train test split.

We then trained a multi a multi-classification CNN for species classification achieving a training accuracy of 90% and testing accuracy of 86.8%. This is also siginificantly higher than our 16.67% accuracy threshold.

We noticed our positive samples are not perfect pixel-for-pixel representations of what we are actually detection, our bird. Our samples are bounding boxes that also bring in random noise. So different species of birds will tend to have different background noise that we will also be classifying on.

We made the realization that the presence of false positives is detrimental to detection. We can use our AUC-ROC curve to reevaluate our probability threshold for predictions and adjust True Positive and False Positive tradeoff for stronger deployment. A higher threshold (>0.9) reduced false positive detection and still provided enough true positives to work with.

For further steps and more time, we can obtain ore data, gridsearch parameters, and use ModelCheckpoints to tune our CNN and deploy it in the setting we are targeting. 

## Sources
- http://www.huppelen.nl/publications/selectiveSearchDraft.pdf
- https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
- https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55
- https://www.pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/
- https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
