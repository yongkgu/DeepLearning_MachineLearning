%% 1. Transfer Learning
%% Overview of Pretrained Convolutional Neural Networks
% Load AlexNet into a network variable in MATLAB

net = alexnet    % Error? Then, install 'Deep Learning Toolbox Model for AlexNet Network' in Add-Ons Explorer
%% 
% Peek into AlexNet

% View the layers of AlexNet and save the layers to a variable
layers = net.Layers

% Save the first and last layers of AlexNet
layer_input = layers(1)
layer_output = layers(end)


% View the output classes of AlexNet
classes = layer_output.Classes
%% Preprocessing an Image
% Import an image and resize it to be compatible with AlexNet

% Display an image
image = imread("harper.jpg")
imshow(image)

% View the input size of AlexNet
layer_input.InputSize

% Resize the image so that it can be applied to AlexNet
image_resized = imresize(image,[227, 227])
imshow(image_resized)

% Classify the image
prediction = classify(net, image_resized)

% Classify the image and see the score
[prediction, score] = classify(net, image_resized)
max(score)

%% Image Datastore
% When you import large data, you don't have to save them as variables because 
% it costs too much time and memory. 
% 
% Instead, you can read image files by creating a datastore, which references 
% a data source such as a folder of image files. When you create a datastore, 
% basic information such as the file name and formats is stored.

% Create an image datastore referring to 'Lucy'
imageDS = imageDatastore("petImages\Lucy\","IncludeSubfolders",true,"LabelSource","foldernames")

% Read and display 'Lucy' images
    % imageAll = imtile(imageDS)
    % imshow(imageAll) 
montage(imageDS)
%% 
% For the images in an image datastore, basic preprocessing can be done by |augmentedImageDatastore| 
% function

augImageDS = augmentedImageDatastore([227 227], imageDS)

%% Modifying a Pretrained Network
% An existing pretrained network does not ouput the classes you want. When performing 
% transfer learning, you will typically change the fully connected layer and the 
% classification layer to suit your specific application. 

    % clear
% Load AlexNet and view the layers and the output classes
net = alexnet;
layers = net.Layers
layer_output = layers(end)
layer_output.Classes

%% 
% Change the number of output classes by modifying the fully conneceted layer(23) 
% and the classification layer(25).

layers(23) = fullyConnectedLayer(14)
layers(end) = classificationLayer();

%% Preparing Training Data

% Create an image datastore with the images labeled by their folder name
imageDS = imageDatastore("petImages","IncludeSubfolders",true,"LabelSource","foldernames");

% Split the data into training and test data sets
[trainImages, testImages] =splitEachLabel(imageDS,0.8) % 80% data를 train data로 사용

% Create augmented datastores to resize the images so that they can be applied to AlexNet
trainData = augmentedImageDatastore([227 227],trainImages); %ALEX NET에 맞게 이미지 RESIZED
testData = augmentedImageDatastore([227 227],testImages);


%% Training Options
% Use Stochastic Gradient Descent with Momentum (SGDM) and decrease the initial 
% learning rate to 0.0001.
% 
% The default initial learning rate is 0.01 but generally it should be decreased 
% for better performances in a transfer learning.

opts = trainingOptions("sgdm","InitialLearnRate",0.0001,"Plots","training-progress")
%% Training and Evaluating the Network
% Train a new network. In order to use a GPU instead of a CPU, install 'Parallel 
% Computing Toolbox' in Add-Ons explorer. A GPU is much faster than a CPU in training 
% a neural network.

newnet = trainNetwork(trainData,layers,opts) 
%"InitialLearnRate" : 작업의 보폭을 설정
% 미니배치 : 가중치 값을 업데이트하는 주기를 결정
% 에폭 : TRAIN 데이터를 30번 학습한다.
%% 
% Evaluate the network by classifying the test data

testPred = classify(newnet, testData); % Prediction by the new network
testGT = testImages.Labels;    % Ground truth of the test data 실제값
nnz(testPred == testGT)/numel(testPred)   % Prediction Accuracy 예측이 맞은 갯수 %numel (number of element)
confusionchart(testGT, testPred)    % Visualization in a confusion matrix