%% COMPARECLASSIFIERS
% Compare the performance of the classifiers for the binary classification problem. 
%% Prepare data

load heartdisease
%% 
% Divide the data into training and validation sets

rng(1234)
part = cvpartition(heartdataNum.HeartDisease,'Holdout',0.2);
tridx = training(part);
tdata = heartdataNum(tridx,:);
vdata = heartdataNum(~tridx,:);
tdataAll = heartdataAll(tridx,:);
vdataAll = heartdataAll(~tridx,:);
%% 
% Create a table to hold the results

mdlnames = {'kNN','kNN k=5','Weighted kNN k=5','Tree','Pruned tree',...
    'Tree (all predictors)','Pruned tree (all predictors)',...
    'NB','NB kernel','NB (all predictors)','NB kernel (all predictors)',...
    'Linear DA','Quadratic DA','Linear SVM','Gaussian SVM',...
    'Linear SVM (all predictors)','Gaussian SVM (all predictors)'};
results = table(zeros(17,1),zeros(17,1),...
    'RowNames',mdlnames,'VariableNames',{'ResubLoss','Loss'});
%% kNN models

m = fitcknn(tdata,'HeartDisease');
results{1,:} = [resubLoss(m) loss(m,vdata)];

m.NumNeighbors = 5;
results{2,:} = [resubLoss(m) loss(m,vdata)];

m.DistanceWeight = 'squaredinverse';
results{3,:} = [resubLoss(m) loss(m,vdata)];
%% Tree models

m = fitctree(tdata,'HeartDisease');
results{4,:} = [resubLoss(m) loss(m,vdata)];

m = prune(m,'Level',3);
results{5,:} = [resubLoss(m) loss(m,vdata)];

m = fitctree(tdataAll,'HeartDisease');
results{6,:} = [resubLoss(m) loss(m,vdataAll)];

m = prune(m,'Level',3);
results{7,:} = [resubLoss(m) loss(m,vdataAll)];
%% Naive Bayes models

m = fitcnb(tdata,'HeartDisease');
results{8,:} = [resubLoss(m) loss(m,vdata)];

m = fitcnb(tdata,'HeartDisease','Distribution','kernel');
results{9,:} = [resubLoss(m) loss(m,vdata)];

m = fitcnb(tdataAll,'HeartDisease');
results{10,:} = [resubLoss(m) loss(m,vdataAll)];

dists = [repmat({'kernel'},1,11),repmat({'mvmn'},1,10)];
m = fitcnb(tdataAll,'HeartDisease','Distribution',dists);
results{11,:} = [resubLoss(m) loss(m,vdataAll)];
%% Discriminant Analysis models

m = fitcdiscr(tdata,'HeartDisease');
results{12,:} = [resubLoss(m) loss(m,vdata)];

m = fitcdiscr(tdata,'HeartDisease','DiscrimType','quadratic');
results{13,:} = [resubLoss(m) loss(m,vdata)];
%% SVM models

m = fitcsvm(tdata,'HeartDisease');
results{14,:} = [resubLoss(m) loss(m,vdata)];

m = fitcsvm(tdata,'HeartDisease','KernelFunction','gaussian');
results{15,:} = [resubLoss(m) loss(m,vdata)];

m = fitcsvm(tdataAll,'HeartDisease');
results{16,:} = [resubLoss(m) loss(m,vdataAll)];

m = fitcsvm(tdataAll,'HeartDisease','KernelFunction','gaussian');
results{17,:} = [resubLoss(m) loss(m,vdataAll)];
%% View results

disp(results)

disp(sortrows(results,'Loss'))