%% 8. Classifying Sequence Data
%% Investigate the Sequences
%     Load the musical instrument data
% This contains 472 recordings of the Flute, Cello, and Piano, stored in cell 
% arrays, along with their correct labels stored in categoricals. There is also 
% a sampling rate of the recordings.

load classify3InstrumentsData.mat
%% 
% Display the class distribution of the training data.

summary(YTrain)
%     Play some samples
% These sequences can be played using the sampling rate |fs| and the function 
% |sound|. Display the corresponding labels.

x =XTrain{1}
y = YTrain(1)
% sound(x,fs)  % 하나의 소리 듣기

idx = randperm(numel(XTrain),3) % 441개의 수중 무작위로 3개의 수 뽑기
x = [XTrain{idx}] % 배열로 감싸면 1개의 인덱스가 아니라 3개의 인덱스가 이어져서 나오게 됨.
y = YTrain(idx)

sound(x,fs) % 3개의 소리 듣기
%     Plot the sound waves

plot(x)
xlabel('Time Steps')
ylabel('Amplitude')

%% Create the Network
% LSTM은 CNN보다 layer구성이 간단해도 충분한 성능을 보장한다.
%     Assemble network layers
%% 
% # Start with a sequence input layer with a single input node, since these 
% sequences only have one features. 
% # Follow with a BiLSTM layer with 100 nodes. Output only the last time step 
% of each recording sample being classified. 
% # End with a fully connected layer with one node for each instrument, a softmax 
% layer, and a classificationLayer.layer
% # 

layers = [
    sequenceInputLayer(1);
    bilstmLayer(100,"OutputMode","last");
    fullyConnectedLayer(3);
    softmaxLayer();
    classificationLayer();
    ]
%% Train the Network
% 파라미터 튜닝과 시퀀셜 렝스가 매우 중요하다.
% 
% AI 대회에서 보통은 layer를 어떻게 구성하는지가 중요하지만, LSTM에서는 파라미터 튜닝이 더 중요하다. 
%     Set training options
%% 
% * Use the Adam optimizer
% * Set the maximum number of epochs to 250 처음에는 여유롭게 에폭을 잡아보는게 좋다.
% * The initial learning rate to 0.005
% * The gradientThreshold to 1
% * Plot the training progress
% * Make it so the learning rate can drop for the first 60 epochs

options = trainingOptions("adam","MaxEpochs",250,"Plots","training-progress",...
    "InitialLearnRate",0.005,"GradientThreshold",1,...
    "LearnRateSchedule","piecewise","LearnRateDropPeriod",60,"LearnRateDropFactor",0.1)
%에폭 60부터 러닝레이트를 0.1 떨구겠다.

% LearnRateDropFactor 와 GradientDecayFactor와의 차이점? 
% 스파이크 = 로스가 확 튀는 현상
% 플럭체이션이 있다.  --> GradientThreshold를 설정해준다. (러닝레이트 올려준다.) -> 너무 높게 잡으면 안됨
% 나아질 여지가 보이면 에폭을 늘리고 러닝레이트를 낮춘다.
%급격히 정확도가 낮아지면 러닝레이트 스케줄을 조절한다.
%     Train the Network 


%% Load the trained Network

load net_LRDP200.mat % 'LearnRateDropPeriod' 200
% load net_LRDP50.mat % 'LearnRateDropPeriod' 50
%% Classify and Evaluate the Network

testPred = classify(net,XTest);
confusionchart(YTest,testPred)