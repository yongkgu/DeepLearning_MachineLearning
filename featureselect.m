%% FEATURESELECT
% Perform feature selection on the heart disease data for a Naive Bayes classifier 
% and an SVM. 
%% Load the data

load heartdisease
%% 
% Extract the response variable

HD = heartdataNum.HeartDisease;
%% 
% Make a partition for evaluation

rng(1234)
part = cvpartition(HD,'KFold',10);
%% Perform sequential feature selection on a Naive Bayes classifier

rng(1234)
fmodel = @(X,y) fitcnb(X,y,'Distribution','kernel');
ferror = @(Xtrain,ytrain,Xtest,ytest) nnz(predict(fmodel(Xtrain,ytrain),Xtest) ~= ytest);
tokeep = sequentialfs(ferror,numdata,HD,'cv',part);
%% 
% Which variables are in the final model?

vars(tokeep)
%% 
% Fit a model with just the given variables

m = fitcnb(numdata(:,tokeep),HD,...
    'Distribution','kernel','CVPartition',part);
kfoldLoss(m)
%% 
% And again with SVM

rng(1234)
fmodel = @(X,y) fitcsvm(X,y);
ferror = @(Xtrain,ytrain,Xtest,ytest) nnz(predict(fmodel(Xtrain,ytrain),Xtest) ~= ytest);
tokeep = sequentialfs(ferror,numdata,HD,'cv',part);
%% 
% Which variables are in the final model?

vars(tokeep)
%% 
% Fit a model with just the given variables

m = fitcsvm(numdata(:,tokeep),HD,'CVPartition',part);
kfoldLoss(m)