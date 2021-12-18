%% 3. Building Classification Models
%% Supervised Learning
% Course Example: Heart Disease

% Load the example data
load('heartdisease.mat')
% Is it necessary to apply a classification model to the data?
% Pre-evaluation of the data
[g, c] = kmeans(zscore(heartdataNum{:,1:11}),2); %그룹을 2개로 나누다 
counts = crosstab(g,heartdataNum.HeartDisease)
figure
bar(counts,'stacked')
legend([{'true'},{'false'}])

% Training and Validation

% Divide the heart disease into training(80%) and validation(20%) sets
rng(1234)
part = cvpartition(heartdataNum.HeartDisease,'HoldOut',0.2) %validation set 20% 
tridx = training(part);
tdata = heartdataNum(tridx,:)
vdata = heartdataNum(~tridx,:)
tdataAll = heartdataAll(tridx,:);
vdataAll = heartdataAll(~tridx,:)

% 1. Nearest Neighbor Classification(1/2)
% The most straightforward way of classification without any assumptions about 
% the underlying distribution of the data

m = fitcknn(tdata,'HeartDisease');
% Why use tdata, not tdataAll? because k-NN does not accept mixed predictors(numeric+categorical).
% Evaluating Classifications

m = fitcknn(tdata,'HeartDisease')
% Measure loss
errKNN_train = resubLoss(m) % Measure loss with the training data , To check Overfiting problem 
errKNN_test = loss(m,vdata)% Measure loss with the test(validation) data

% Visualize prediction accuracy
HDpred = predict(m,vdata)
HDtrue = vdata.HeartDisease;
figure
confusionchart(HDtrue,HDpred) % blue is coreect, red is incorrect 




% 1. Nearest Neighbor Classification(2/2)

% Check overfitting and Update model 1
m = fitcknn(tdata,'HeartDisease',"NumNeighbors",5)
errKNN_train = resubLoss(m) 
errKNN_test = loss(m,vdata)
HDpred = predict(m,vdata);
HDtrue = vdata.HeartDisease;
figure
confusionchart(HDtrue,HDpred)
 
 
% Update model 2
m.DistanceWeight = 'squaredinvers';
errKNN_train = resubLoss(m) 
errKNN_test = loss(m,vdata)


% 2. Decision Tree Classification
% A decision tree model make a categorical response prediction by following 
% a sequence of binary decisions based on predictor variable value. It does not 
% assume any underlying distributions of the data like k-NN.
% 
% A decision tree model is robust to noise data.

% Fit and evaluate a decision tree model with default settings
m = fitctree(tdata, 'HeartDisease');
%view(m,'Mode','graph');

errDT_train = resubLoss(m)
errDT_test = loss(m,vdata)
HDpred = predict(m,vdata);
HDtrue = vdata.HeartDisease;
figure
confusionchart(HDtrue,HDpred)

% Check overfitting and update model with 'prune' function
m = prune(m,'Level',3);
view(m,'Mode','graph')
errPruneDT_train = resubLoss(m)
errPruneDT_test = loss(m,vdata)

% Unlike k-NN, a decision tree model accepts mixed data(numeric + categorical)
m = fitctree(tdataAll, 'HeartDisease')
errDT_train = resubLoss(m)
errDT_test = loss(m,vdataAll)

m = prune(m,'Level',3);
errPruneDT_train = resubLoss(m)
errPruneDT_test = loss(m,vdataAll)

% 3. Naive Bayes Classification
% A Naive Bayes model uses Bayes's rule of conditional probability to estimate 
% the probability of a given observation being in each response class. Accordingly, 
% it assumes an underlying distribution of the data. (Note: Each predictor variable 
% should be independent, If not, use a 'Discriminant Analysis')

m = fitcnb(tdata,'HeartDisease')
errNB_train = resubLoss(m)
errNB_test = loss(m,vdata) 

% Use kernel smoothing instead of normal distributions
% Kernel smoothing only works for numeric predictor variables
m = fitcnb(tdata,'HeartDisease','DistributionNames',"kernel")
errNB_train = resubLoss(m)
errNB_test = loss(m,vdata)



% Use a mix of numeric and categorical predictors and use kernel smoothing for all numerical predictors
m = fitcnb(tdataAll, 'HeartDisease');
errNB_train = resubLoss(m)
errNB_test = loss(m,vdataAll) 

dists = [repmat({'kernel'},1,11),repmat({'mvmn'},1,10)]
%dists2 = repmat({'mvmn'},1,10)
m = fitcnb(tdataAll, 'HeartDisease','DistributionNames',dists)
errNB_train = resubLoss(m)
errNB_test = loss(m, vdataAll)