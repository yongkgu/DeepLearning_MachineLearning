%% Improving Predictive Models
%% 3. Reducing Predictors
% 3.1. Feature transformation
% Use PCA to reduce the number of predictors in a Naive Bayes classifier.

load heartdisease
HD = heartdataNum.HeartDisease; % Extract the response variable
numdata = heartdataNum{:,1:end-1}; % Extract the numeric predictors as an array
vars = heartdataNum.Properties.VariableNames(1:end-1); % Extract the predictor names
%% 
% Perform PCA

[pcs,scrs,~,~,pexp] = pca(numdata);
pareto(pexp) % You can find that the first eight principal components explain over 90% of variance
%% 
% Fit a Naive Bayes Model on the reduced data

rng(1234)
printCom8 = scrs(:,1:8) % Use the first eight principal components
part = cvpartition(HD,'KFold',10);

% Fit and evaluate the model on the full dataset.
mdlFull = fitcnb(numdata,HD,'DistributionNames',"kernel",'CVPartition',part);
fullDataLoss = kfoldLoss(mdlFull)

% Fit and evaluate the model on the reduced dataset.
mdlReduced = fitcnb(printCom8,HD,'DistributionNames','kernel','CVPartition',part);
reduceDataLoss = kfoldLoss(mdlReduced)

%% 
% Visualize the relationship between the variables and the principal components
%% 
% # biplot 
% # heatmap (useful) - very useful
% # parallelcoords 
% # factor analysis

% Verify that the principal components are uncorrelated
%figure
%plotmatrix(scrs)

% 1.Visualize the relationship using biplot function
figure 
biplot(pcs(:,1:2),'VarLabels',vars)


% 2.Visualize the relationship using heatmap function - important
figure
heatmap(abs(pcs(:,1:8)),'YDisplayLabels',vars)
xlabel('Principal Components')

% 3. Visualize predictive power of the principal components
figure
parallelcoords(scrs,'Group',HD,'Quantile',0.25)


% 4. Factor Analysis
[Lambda,Psi,~,~,E] = factoran(numdata,6);
heatmap(abs(Lambda),'YDisplayLabels',vars)
% 3.2. Feature Selection
% 1. Built-in feature selection for decision tree models

open('treeselect.mlx')
%% 
% *2. Sequential Feature Selection (IMPORTANT!)*
% 
% Perform sequential feature selection on the heart disease data for Naive Bayes 
% and SVM classifiers.

rng(1234)
part = cvpartition(HD,'KFold',10);
%% 
% Sequential feature selection on a Naive Bayes classifier.

rng(1234)
tokeep_nb = sequentialfs(@errorNB,numdata,HD,'cv',part);
vars(tokeep_nb)
%% 
% Compare the results with the full dataset.

% Naive Bayes with the full data set.
ndlFull = fitcnb(numdata,HD,'DistributionNames',"kernel",'CVPartition',part)
fullDataLoss = kfoldLoss(ndlFull)

% Naive Bayes with the selected predictors.
numdataSelectedNB = numdata(:,tokeep_nb)
ndlSelected = fitcnb(numdataSelectedNB,HD,'DistributionNames','kernel','CVPartition',part);
selectedDataLoss = kfoldLoss(ndlSelected)


%% 
% Sequential feature selection on a SVM classifier.

rng(1234)
tokeep_svm = sequentialfs(@errorSVM,numdata,HD,'cv',part)
vars(tokeep_svm)

%% 
% Compare the results with the full data set.

rng(1234)
% SVM with full data set.
ndlFull = fitcsvm(numdata,HD,'CVPartition',part);
fullDataLoss = kfoldLoss(ndlFull)

% SVM with the selected predictors.
numdataSelectedSVM = numdata(:,tokeep_svm);
ndlSelected = fitcsvm(numdataSelectedSVM,HD,'CVPartition',part);
selectedDataLoss = kfoldLoss(ndlSelected)


%% 
% Callback function: errorNB function returns the number of inaccurate predictions 
% for |ytest| by a Naive Bayes classifier.

function error = errorNB(Xtrain,ytrain,Xtest,ytest)
mdlnb = fitcnb(Xtrain , ytrain,"DistributionNames","kernel")
ypred = predict(mdlnb,Xtest)
error = nnz(ypred ~= ytest)
end

%% 
% Callback function: errorSVM function returns the number of inaccurate predictions 
% for |ytest| by a SVM classifier.

function error = errorSVM(Xtrain,ytrain,Xtest,ytest)
mdlsvm = fitcsvm(Xtrain,ytrain);
ypred = predict(mdlsvm,Xtest);
error = nnz(ypred ~= ytest);
end