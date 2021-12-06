%% 5. Improving Network Performance
%%   5.2. Training Option
%% Load Dataset

load("Dataset_reshaped.mat")
%%
montage(test_x)
%%
test_y
categories(test_y)
a = find(test_y == categorical(0)) %인덱스 넘버
b = test_y == categorical(0) % 참, 거짓
%%
size(test_x)
%% Spliting Data
% Spliting training data into training and validation sets. (8:2)

pt = cvpartition(train_y,"HoldOut",0.2)

val_x = train_x(:,:,:,pt.test); %val데이터니까 pt2
val_y = train_y(pt.test);

training_x = train_x(:,:,:,pt.training);
training_y = train_y(pt.training);

%% Creating Network
% Create a Convolutional Neural Network.

layers = [imageInputLayer([28,28,1]);convolution2dLayer([3,3],20);reluLayer(); maxPooling2dLayer([3,3]);fullyConnectedLayer(10);softmaxLayer();classificationLayer()];

layers2 = [imageInputLayer([28,28,1]);convolution2dLayer([2,2],20);reluLayer(); maxPooling2dLayer([2,2]);convolution2dLayer([2,2],40);reluLayer(); maxPooling2dLayer([2,2]);convolution2dLayer([2,2],80);reluLayer(); maxPooling2dLayer([2,2]);fullyConnectedLayer(10);softmaxLayer();classificationLayer()];




%% Training Network
% Create appropriate training options and train the network

option1 = trainingOptions("sgdm","MaxEpochs",50,"InitialLearnRate",0.0001,"Plots","training-progress","ValidationData",{val_x,val_y})
option2 =  trainingOptions("sgdm","MaxEpochs",30,"InitialLearnRate",0.1,"GradientThreshold",1,"Plots","training-progress","ValidationData",{val_x,val_y},...
    "LearnRateSchedule","piecewise","LearnRateDropPeriod",5);

newNet = trainNetwork(training_x,training_y,layers,option1);




%% Evaluating Network
% Calculate the prediction accuracy on the test data.

testPred = classify(newNet,test_x);
testGT = test_y;
testAcc = nnz(tesetPred == testGT) / numel(testPred)
confusionchart(test_y,testPred)