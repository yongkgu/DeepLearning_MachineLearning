%% FUELSTEPWISE
% Fit and evaluate linear models of fuel economy using stepwise fitting.
%% Load data

load carEcon
%% Perform a stepwise linear fit of fuel economy

mdl = stepwiselm(carTrain);
%% Evaluate model at test predictor values

resppred = predict(mdl,carTest);
%% Compare predicted and actual responses

evaluatefit(carTest.FuelEcon,resppred,'Stepwise')
%% Change criterion and limit to linear models

mdl = stepwiselm(carTrain,'Criterion','aic','Upper','linear');
resppred = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,resppred,'Stepwise with AIC')