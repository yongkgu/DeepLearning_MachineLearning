%% TWOPREDICTORS
% Fit and evaluate a linear model with one numeric and one categorical predictor.
%% Load data

load carEcon
%% Fit a 2-predictor linear model of fuel economy

figure
scatter3(carData.EngDisp,carData.City_Highway,carData.FuelEcon,'.')
mdl = fitlm(carTrain(:,{'EngDisp','City_Highway','FuelEcon'}));
%% Add model to plot

x = [min(carData.EngDisp);min(carData.EngDisp);max(carData.EngDisp);max(carData.EngDisp)];
y = categorical({'city';'highway';'city';'highway'});
T = table(x,y,'VariableNames',{'EngDisp','City_Highway'});
z = predict(mdl,T);
x = reshape(x,2,2);
y = reshape(y,2,2);
z = reshape(z,2,2);
hold on
surf(x,y,z)
xlabel('Engine displacement (L)')
ylabel('City [1]/Highway [2]')
zlabel('Fuel economy (L/100km)')
%% Evaluate model at test predictor values

econpred = predict(mdl,carTest(:,{'EngDisp','City_Highway'}));
%% Compare predicted and actual responses

evaluatefit(carTest.FuelEcon,econpred,'linear (disp + city/hwy)')