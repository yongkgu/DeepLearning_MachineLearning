%% CROSSVALMODELS
% Use 10-fold cross validation to evaluate the various predictive models for 
% the heart disease data.
%% Run comparison with holdout validation

compareclassifiers
%% Extract results
% Add variable to hold cross validation errors

results.kFoldLoss = zeros(height(results),1);
%% 
% Remove pruned trees (can't cross-validate)

results([5 7],:) = [];
%% Make a cross validation partition

rng(1234)
part = cvpartition(heartdataNum.HeartDisease,'KFold',10);
%% Create cross-validated models
% Note that the cross-validated models are only used to evaluate performance. 
% The fitted models do not have a |predict| method.
% 
% _*k-_NN models*

m = fitcknn(heartdataNum,'HeartDisease','CVPartition',part);
results.kFoldLoss(1) = kfoldLoss(m);

m = fitcknn(heartdataNum,'HeartDisease','NumNeighbors',5,'CVPartition',part);
results.kFoldLoss(2) = kfoldLoss(m);

m = fitcknn(heartdataNum,'HeartDisease','NumNeighbors',5,...
    'DistanceWeight','squaredinverse','CVPartition',part);
results.kFoldLoss(3) = kfoldLoss(m);
%% 
% *Tree models*

m = fitctree(heartdataNum,'HeartDisease','CVPartition',part);
results.kFoldLoss(4) = kfoldLoss(m);

m = fitctree(heartdataAll,'HeartDisease','CVPartition',part);
results.kFoldLoss(5) = kfoldLoss(m);
%% 
% *Naive Bayes models*

m = fitcnb(heartdataNum,'HeartDisease','CVPartition',part);
results.kFoldLoss(6) = kfoldLoss(m);

m = fitcnb(heartdataNum,'HeartDisease','Distribution','kernel','CVPartition',part);
results.kFoldLoss(7) = kfoldLoss(m);

m = fitcnb(heartdataAll,'HeartDisease','CVPartition',part);
results.kFoldLoss(8) = kfoldLoss(m);

dists = [repmat({'kernel'},1,11),repmat({'mvmn'},1,10)];
m = fitcnb(heartdataAll,'HeartDisease','Distribution',dists,'CVPartition',part);
results.kFoldLoss(9) = kfoldLoss(m);
%% 
% *Discriminant Analysis models*

m = fitcdiscr(heartdataNum,'HeartDisease','CVPartition',part);
results.kFoldLoss(10) = kfoldLoss(m);

m = fitcdiscr(heartdataNum,'HeartDisease','DiscrimType','quadratic','CVPartition',part);
results.kFoldLoss(11) = kfoldLoss(m);
%% 
% *SVM models*

m = fitcsvm(heartdataNum,'HeartDisease','CVPartition',part);
results.kFoldLoss(12) = kfoldLoss(m);

m = fitcsvm(heartdataNum,'HeartDisease','KernelFunction','gaussian','CVPartition',part);
results.kFoldLoss(13) = kfoldLoss(m);

m = fitcsvm(heartdataAll,'HeartDisease','CVPartition',part);
results.kFoldLoss(14) = kfoldLoss(m);

m = fitcsvm(heartdataAll,'HeartDisease','KernelFunction','gaussian','CVPartition',part);
results.kFoldLoss(15) = kfoldLoss(m);
%% View results

disp(results)

disp(sortrows(results,'kFoldLoss'))