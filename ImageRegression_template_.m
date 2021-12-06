%% 6. Image Regression

load imageRegressionData.mat
%% What is Regression?
% Regression is another task that can be accomplished with deep learning. _Regression_ 
% refers to assigning continuous response values to data, instead of discrete 
% classes.
% 
% One example of image regression is correcting rotated images. The input data 
% is a rotated image, and the known response is the angle of rotation.
% 
% 
% 
% You will build a network that corrects the color in images. 
% 
% A set of images have been modified by changing their red, green, and blue 
% channels. The response for each modified image is three numeric values that 
% correspond to the intensity increase or decrease of the corresponding channel.
% Peek into the data

trainingData % 파일 위치, r, g, b
imshow(imread(trainingData.File{1})) % 셀 형태로 추출해서 디스플레이
size(imread(trainingData.File{1}))
%% Transfer Learning for Image Regression
%     Modify the network layers for regression
%% 
% # Replace the |fullyConnectedLayer| 
% # Delete the softmax layer and the classification layer
% # Add a |regressionLayer|

net = alexnet;
layers = net.Layers
analyzeNetwork(net)
%     Prepare the data

trainds = augmentedImageDatastore([227,227], trainingData);
testds = augmentedImageDatastore([227,227], testData);
%%
layers(23) = fullyConnectedLayer(3); %fully co layer % 연속된 숫자의 경우에는 2
layers(24:end) = []; % 계층 삭제
layers = [layers;regressionLayer] % 회귀출력 만들어주기 (여러번 실행 주의!)

%% Train the Network 

% 빛 바랜 이미지 복구, 화질 개선 등에 이미지 regression이 사용 됨.
options = trainingOptions("adam","MaxEpochs",10,"InitialLearnRate",0.0001,...
    "Plots","training-progress");

newNet = trainNetwork(trainds,layers,options)

% 평소에는 val 데이터 추가해서 어디까지 에폭 설정해도 되는지 모니터링하면서 파라미터 수정할 것.
%% Evaluate the Network
% Predict the response for all images in the test data and calculate the root 
% mean squre error (RMSE) for the test data set.

testPred = predict(newNet, testds);
rgbGT = testData.Color;
err = rgbGT - testPred;
rmse = sqrt(mean(err.^2))




%% Correct the color
% Use the network to correct the rgb value of the first test image and display 
% the corrected image.

imageNum = 1;

testImage = imread( testData.File{imageNum})
imshow(testImage)

%첨부된 함수 사용
rgb = testPred(imageNum,:) %첫 번째 이미지의 예측값
correctedImage = correctColor(testImage,rgb) % rgb 업데이트 (색 보정)
imshow(correctedImage)
%%