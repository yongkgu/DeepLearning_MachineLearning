%% FUELTREE
% Fit and evaluate decision tree models of fuel economy.
%% Load data

load carEcon
%% Fit a decision tree regression model of fuel economy

mdl = fitrtree(carTrain,'FuelEcon');
%% Evaluate model at test predictor values

resppred = predict(mdl,carTest);
%% Compare predicted and actual responses

evaluatefit(carTest.FuelEcon,resppred,'Tree')
%% Prune the tree

mdl = prune(mdl,'Level',10);
resppred = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,resppred,'Pruned tree')
%% Set minimum leaf size

mdl = fitrtree(carTrain,'FuelEcon','MinLeafSize',5);
resppred = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,resppred,'Tree with leaf limit')