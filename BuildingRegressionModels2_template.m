%% Building Regression Models - Nonparametric Models
%% 3. SVMs and Decision Trees
% Course Example: Fuel Economy
% Load fuel economy data and view the relationships between the numeric predictors 
% and response.

load carEcon
carData
% 3.1. Fit a SVM predictor linear model of fuel economy

mdl = fitrsvm(carTrain,'FuelEcon')
%% 
% Predict fuel economy for the test data and compare the result with the actual 
% fuel economy of the test data.

econPred = predict(mdl, carTest);
evaluatefit(carTest.FuelEcon,econPred,'Linear SVM')
%% 
% Standardize variables

mdl = fitrsvm(carTrain, 'FuelEcon','Standardize',true);
econPred = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,econPred,'Standardize Linear SVM')
%% 
% Try a Gaussian kernel SVM

mdl = fitrsvm(carTrain, 'FuelEcon','KernelFunction','gaussian');
econPred = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,econPred,'Gaussian SVM')
% 3.2. Fit a Decision Tree linear model of fuel economy

mdl = fitrtree(carTrain,'FuelEcon');
%% 
% Predict fuel economy for the test data and compare the result with the actual 
% fuel economy of the test data.

econPred = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,econPred,'Decision Tree')
%% 
% Prune the tree

mdl = prune(mdl, 'Level',10);
econPred = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,econPred,'Pruned Tree')
%% 
% Set minimum leaf size

mdl = fitrtree(carTrain,'FuelEcon','MinLeafSize',5);
econPred = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,econPred,'Tree with leaf limit')
%% 4. Gaussian Process Regression
% Fit and evaluate Gaussian process models of fuel economy. - important

mdl = fitrgp(carTrain,'FuelEcon');
[econPred, econStd, econInt] = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,econPred,'GPR')
subplot(2,2,1) % Add prediction intervals
hold on
plot(econInt,'k:')
 
%% 
% Try a different kernel - Matern

mdl = fitrgp(carTrain,'FuelEcon','KernelFunction','matern32');
[econPred, econStd, econInt] = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,econPred,'GPR (Matern kernel')
subplot(2,2,1) % Add prediction intervals
hold on
plot(econInt,'k:')
 
 
 
%% 
% Try a different kernel - Automatic Relevance Determination Squared Exponential
% 
% You can choose a kernel function which uses a separate length scale for each 
% predictor. Such kernel functions have names which start with |"ard".|

tic
mdl = fitrgp(carTrain,'FuelEcon','KernelFunction','ardsquaredexponential');
[econPred, econStd, econInt] = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,econPred,'GPR (Ardsquaredexponential kernel')
subplot(2,2,1) % Add prediction intervals
hold on
plot(econInt,'k:')
toc