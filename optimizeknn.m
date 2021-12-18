%% OPTIMIZEKNN
% Use hyperparameter optimization to automatically tune _k_-NN models for the 
% heart disease data.
%% Load data

load heartdisease
%% Divide the data into training and validation sets

rng(1234)
part = cvpartition(heartdataNum.HeartDisease,'Holdout',0.2);
tridx = training(part);
dataTrain = heartdataNum(tridx,:);
dataTest = heartdataNum(~tridx,:);
%% Train a _k_-NN model with default properties

m = fitcknn(dataTrain,'HeartDisease');
resubLoss(m)
loss(m,dataTest)
%% Turn on hyperparameter optimization
% Setting '|OptimizeHyperparamters|' to '|auto|' will optimize these _k_-NN 
% properties:
%% 
% * |NumNeighbors|
% * |Distance|

m = fitcknn(dataTrain,'HeartDisease','OptimizeHyperparameters','auto');
%% 
% The command-line output contains information on the evaluated property values 
% and the hyperparameters chosen for the final model.

m.Distance
m.NumNeighbors
%% 
% The first plot shows the best objective function against the iteration number. 
% The second plot is a model of the objective function against the parameters. 
% You can hide these plots with the '|ShowPlots|' argument.

resubLoss(m)
loss(m,dataTest)
%% Change optimization options
% Make cross validation partition

part = cvpartition(dataTrain.HeartDisease,'KFold',10);
%% 
% Change the validation to use '|part|' and decrease the number of objective 
% evaluations.

opt = struct('CVPartition',part,'MaxObjectiveEvaluations',20);
%% 
% Setting '|OptimizeHyperparamters|' to '|all|' will optimize these _k_-NN properties:
%% 
% * |NumNeighbors|
% * |Distance|
% * |DistanceWeight|
% * |Exponent|
% * |Standardize|

m = fitcknn(dataTrain,'HeartDisease','OptimizeHyperparameters','all','HyperparameterOptimizationOptions',opt);
%% 
% Note that there is only one plot displayed. You will only see the second plot 
% if there are one or two optimization parameters, and if '|Optimizer|' is set 
% to '|bayesopt|'.

resubLoss(m)
loss(m,dataTest)