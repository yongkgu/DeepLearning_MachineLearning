%% FUELGPR
% Fit and evaluate Gaussian process models of fuel economy.
%% Load data

load carEcon
%% Fit a Gaussian process model of fuel economy

mdl = fitrgp(carTrain,'FuelEcon');
%% Evaluate model at test predictor values

[econpred,~,epint] = predict(mdl,carTest);
%% Compare predicted and actual responses

evaluatefit(carTest.FuelEcon,econpred,'GPR')
%% 
% Add intervals to predictions

subplot(2,2,1)
hold on
plot(epint,'k:')
%% Try a different kernel

mdl = fitrgp(carTrain,'FuelEcon','KernelFunction','matern52');
[econpred,~,epint] = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,econpred,'GPR (Matern kernel)')
subplot(2,2,1)
hold on
plot(epint,'k:')