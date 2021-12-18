%% PCAREDUCE
% Use PCA to reduce the number of predictors in a Naive Bayes classifier.
%% Load the data

load heartdisease
%% 
% Extract the response variable

HD = heartdataNum.HeartDisease;
%% Perform PCA

[pcs,scrs,~,~,pexp] = pca(numdata);
pareto(pexp)
%% Fit a Naive Bayes model on the reduced data

numdatreduced = scrs(:,1:9);
%% 
% Make a partition for evaluation

rng(1234)
part = cvpartition(HD,'KFold',10);
%% 
% Fit and evaluate the model on the full data set

m = fitcnb(numdata,HD,'Distribution','kernel','CVPartition',part);
kfoldLoss(m)
%% 
% Fit and evaluate the model on the reduced data set

m = fitcnb(numdatreduced,HD,'Distribution','kernel','CVPartition',part);
kfoldLoss(m)