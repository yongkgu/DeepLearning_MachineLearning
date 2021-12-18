%% FUELSVM
% Fit and evaluate SVM models of fuel economy.
%% Load data

load carEcon
%% Fit an SVM regression model of fuel economy

mdl = fitrsvm(carTrain,'FuelEcon');
%% Evaluate model at test predictor values

resppred = predict(mdl,carTest);
%% Compare predicted and actual responses

evaluatefit(carTest.FuelEcon,resppred,'Linear SVM')
%% Standardize variables

mdl = fitrsvm(carTrain,'FuelEcon','Standardize',true);
resppred = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,resppred,'Standardized Linear SVM')
%% Try a different kernel

mdl = fitrsvm(carTrain,'FuelEcon','KernelFunction','gaussian');
resppred = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,resppred,'Gaussian SVM')