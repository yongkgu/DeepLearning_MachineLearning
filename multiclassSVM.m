%% MULTICLASSSVM
% Create and test a multiclass SVM classifier.
%% Prepare data

load heartdisease
%% 
% Divide the data into training and validation sets

rng(1234)
part = cvpartition(heartdataNumMulti.HeartDisease,'Holdout',0.2);
tridx = training(part);
tdata = heartdataNumMulti(tridx,:);
vdata = heartdataNumMulti(~tridx,:);
tdataAll = heartdataAllMulti(tridx,:);
vdataAll = heartdataAllMulti(~tridx,:);
%% Linear SVM

template = templateSVM;
m = fitcecoc(tdata,'HeartDisease','Learners',template);
resubLoss(m)
loss(m,vdata)
%% Gaussian SVM

template = templateSVM('KernelFunction','gaussian');
m = fitcecoc(tdata,'HeartDisease','Learners',template);
resubLoss(m)
loss(m,vdata)
%% Mixed predictors
% Linear

template = templateSVM;
m = fitcecoc(tdataAll,'HeartDisease','Learners',template);
resubLoss(m)
loss(m,vdataAll)
%% 
% Gaussian

template = templateSVM('KernelFunction','gaussian');
m = fitcecoc(tdataAll,'HeartDisease','Learners',template);
resubLoss(m)
loss(m,vdataAll)