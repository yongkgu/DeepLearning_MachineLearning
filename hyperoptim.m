%% HYPEROPTIM
% Compare the accuracy of various models trained with and without hyperparameter 
% optimization.
% 
% Note that this script takes about 5 minutes to run.
%% Load data

load heartdisease
%% Divide the data into training and validation sets

rng(1234)
part = cvpartition(heartdataNum.HeartDisease,'Holdout',0.2);
tridx = training(part);
tdata = heartdataNum(tridx,:);
vdata = heartdataNum(~tridx,:);
%% Create a table to hold the results

none_ResubLoss = zeros([5,1]);
none_Loss = zeros([5,1]);
auto_ResubLoss = zeros([5,1]);
auto_Loss = zeros([5,1]);
mdlnames = {'kNN';'Tree';'Naive Bayes';'Discriminant Analysis';'SVM'};
results = table(none_ResubLoss,none_Loss,auto_ResubLoss,auto_Loss);
results.Properties.RowNames = mdlnames;
%% _k-_NN models

m = fitcknn(tdata,'HeartDisease');
results.none_ResubLoss(1) = resubLoss(m);
results.none_Loss(1) = loss(m,vdata);
%% 
% Setting '|OptimizeHyperparamters|' to 'auto' will optimize these _k_-NN properties:
%% 
% * |NumNeighbors|
% * |Distance|

m = fitcknn(tdata,'HeartDisease','OptimizeHyperparameters','auto');
results.auto_ResubLoss(1) = resubLoss(m);
results.auto_Loss(1) = loss(m,vdata);
%% Tree models

m = fitctree(tdata,'HeartDisease');
results.none_ResubLoss(2) = resubLoss(m);
results.none_Loss(2) = loss(m,vdata);
%% 
% Setting '|OptimizeHyperparamters|' to 'auto' will optimize these tree properties:
%% 
% * |MinLeafSize|

m = fitctree(heartdataNum,'HeartDisease','OptimizeHyperparameters','auto');
results.auto_ResubLoss(2) = resubLoss(m);
results.auto_Loss(2) = loss(m,vdata);
%% Naive Bayes models

m = fitcnb(tdata,'HeartDisease');
results.none_ResubLoss(3) = resubLoss(m);
results.none_Loss(3) = loss(m,vdata);
%% 
% Setting '|OptimizeHyperparamters|' to 'auto' will optimize these Naive Bayes 
% properties:
%% 
% * |Width|
% * |DistributionNames|

m = fitcnb(heartdataNum,'HeartDisease','OptimizeHyperparameters','auto');
results.auto_ResubLoss(3) = resubLoss(m);
results.auto_Loss(3) = loss(m,vdata);
%% Discriminant Analysis models

m = fitcdiscr(tdata,'HeartDisease');
results.none_ResubLoss(4) = resubLoss(m);
results.none_Loss(4) = loss(m,vdata);
%% 
% Setting '|OptimizeHyperparamters|' to 'auto' will optimize these Discriminant 
% Analysis properties:
%% 
% * |Delta|
% * |Gamma|

m = fitcdiscr(heartdataNum,'HeartDisease','OptimizeHyperparameters','auto');
results.auto_ResubLoss(4) = resubLoss(m);
results.auto_Loss(4) = loss(m,vdata);
%% SVM models

m = fitcsvm(tdata,'HeartDisease');
results.none_ResubLoss(5) = resubLoss(m);
results.none_Loss(5) = loss(m,vdata);
%% 
% Setting '|OptimizeHyperparamters|' to 'auto' will optimize these SVM properties:
%% 
% * |BoxConstraint|
% * |KernelScale|

m = fitcsvm(heartdataNum,'HeartDisease','OptimizeHyperparameters','auto');
results.auto_ResubLoss(5) = resubLoss(m);
results.auto_Loss(5) = loss(m,vdata);
%% View results

disp(results)