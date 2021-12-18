%% Ensemble Learning

load heartdisease
HD = heartdataAll.HeartDisease;

rng(1234)
part = cvpartition(HD,'KFold',10);
%% 
% Convert categorical variables to numeric dummy variables

[heartdataAll_dum , predictors_dum] = cattable2mat(heartdataAll)
%% 1. Build an ensemble of 'bagging' trees
% Bag: Bootstrapped aggregation, one of the popular methods along with 'Boosting'.
% 
% For various methods and their supported problems, see the documentation below.

doc ensemble algorithms
%% 
% A single tree

rng(1234)
mdlFulltree = fitctree(heartdataAll_dum,HD,'CVPartition',part);
fullDataLoss = kfoldLoss(mdlFulltree)

%% 
% An ensemble trees

mdlensembletree = fitcensemble(heartdataAll_dum,HD,'Method',"Bag","NumLearningCycles",100,'CVPartition',part);
fullensembleDataLoss = kfoldLoss(mdlensembletree)

%% 2. Build an ensemble of 'Subspace' k-NN
% A single kNN with options

rng(1234)
mdlfullKNN = fitcknn(heartdataAll_dum,HD,'NumNeighbors',5,'Distance',"euclidean",'CVPartition',part);
KNNLoss = kfoldLoss(mdlfullKNN)
%% 
% An ensemble kNN classifiers with options. For the options, use a learner's 
% template.

knnmodel = templateKNN('NumNeighbors',5,'Distance',"euclidean")
mdlEnsembleKNN = fitcensemble(heartdataAll_dum,HD,'Method',"Subspace",'Learners',knnmodel,...
    'NumLearningCycles',100,'CVPartition',part);
EnsembleDataLoss = kfoldLoss(mdlEnsembleKNN)
%% 
%