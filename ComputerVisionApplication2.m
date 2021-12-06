%% 7. Computer Vision Application - Object Detection(2/2)

load('petGT.mat')
%% Train a R-CNN (Regions with Convolutional Neural Networks)
% You can train three different types of R-CNNs : R-CNN, Fast R-CNN, and Faster 
% R-CNN. The corresponding functions are 
%% 
% * |trainRCNNObjectDetector|
% * |trainFastRCNNObjectDetector|
% * |trainFasterRCNNObjectDectector|
%% 
% These networks differ between training time and dectection time. For example, 
% a R-CNN can be trained quickly, but the time to detect a new image is slower 
% than a Faster R-CNN network.
% 
% Choice of which network to use depends on your application. The Fast and Faster 
% R-CNNs are designed to improve detection performance with a large number of 
% regions at the cost of training time.
% NOTE: When you train R-CNNs, the Mini Batch Size should be 1.

opts = trainingOptions("sgdm","InitialLearnRate",0.0001,"MaxEpochs",5,"MiniBatchSize",1); 
% 미니 배치 사이즈 반드시 1 
detector = trainFasterRCNNObjectDetector(petGT,"alexnet",opts);
%% 
% It take long to train a Faster R-CNN using a CPU. Use a trained detector instead.

load detector.mat %이미 학습된 네트워크
%% Use an Object Detector
%     Detect Obejcts
% This code uses |detector| to detect pet faces in |dogim|.
% 
% |[bboxes,scores,labels] = detect(detector,dogim)|

dogim = imread("Ginny.jpg");
imshow(dogim)
% Detect Ginny in a test image.
[bboxes, scores, labels] = detect(detector,dogim)
%     Insert bounding boxes and labels
% The output label from detect is a categorical array. To use this label with 
% |insertObjectAnnotation|, you need to convert it to text. You can use the |cellstr| 
% function to do this. 

detectedDogs = insertObjectAnnotation(dogim,"rectangle",bboxes,labels);
imshow(detectedDogs)
%     Insert bounding boxes, labels and scores
% You can see the bounding boxes and the labels inserted in the image but sometimes, 
% you need to insert the scores in the image as well.

% 카테고리컬이나 셀이나 동일하게 취급해도 무방
labelScores = cell(size(labels))
for k = 1:length(labels)
    labelScores{k} = [char(labels(k)),': ', num2str(scores(k))]
end
detectedDogs = insertObjectAnnotation(dogim,"rectangle",bboxes,labelScores);
imshow(detectedDogs)

%% Threshold Detections
% You may want to apply a threshold the results to only include confident predictions. 
% Extract detections above 90% confidence.

idx = scores > 0.9
bbox = bboxes(idx,:)
labelScore = labelScores(idx);
detectedDogs = insertObjectAnnotation(dogim,"rectangle",bbox,labelScore);
imshow(detectedDogs)