%% Improving Predictive Models
%% 4. Creating Dummy Variables

load heartdisease
HD = heartdataAll.HeartDisease;

rng(1234)
part = cvpartition(HD,'KFold',10);
%% 
% Convert categorical variables to numeric dummy variables

categories(heartdataAll.BloodSugar);
dummyvar(heartdataAll.BloodSugar);

[X, Xnames] = cattable2mat(heartdataAll)
% 4.1. Naive Bayes Classifier with numeric & categorical predictors
% Fit a Naive Bayes classifier to the full data

dist = [repmat({'kernel'},1,11) repmat({'mvmn'},1,10)]
mdl = fitcnb(heartdataAll,'HeartDisease','DistributionNames',dist,'CVPartition',part);
mdlDataLoss = kfoldLoss(mdl)

%% 
% Sequential Feature Selection on a Naive Bayes classifier (Note: Sequentialfs 
% requires predictors in the form of a numeric matrix)

rng(1234)
%error = @(Xtrain, ytrain, Xtest, ytest)...
%    nnz(predict(fitcnb(Xtrain, ytrain,'DistributionNames',"kernel"),Xtest)~= ytest);
%tokeep = sequentialfs(error,X,HD,'cv',part,'options',statset('Display',"iter"))

selectedPredictor = X(:,tokeep)
Xnames(tokeep)
mdlselected = fitcnb(selectedPredictor,HD,'DistributionNames',"kernel","CVPartition",part);
selectedDataLoss = kfoldLoss(mdlselected)

% 4.2. SVM Classifier with numeric & categorical predictors
% Fit a SVM classifier to the full data

mdl = fitcsvm(heartdataAll,'HeartDisease',"KernelFunction","gaussian");
mdlDataLoss = kfoldLoss(mdl)

%% 
% Sequential Feature Selection on a SVM classifier (Note: Sequentialfs requires 
% predictors in the form of a numeric matrix)