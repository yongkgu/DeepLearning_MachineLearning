%% Building Regression Models
%% 1. Linear Regression Models
% Which of these is NOT a linear regression model? (5, 6)
% 
% $$\begin{array}{l}1\ldotp \;y=a+\textrm{bx}+cx^2 \\2\ldotp \;y=ax_1 +{\textrm{bx}}_2 
% +{\textrm{cx}}_3 \\3\ldotp \;y=ax_1 x_2 +bx_2 x_3 +cx_1 x_3 \\4\ldotp \;y=a+\textrm{bsin}\left(x_1 
% \right)+\textrm{csin}\left(x_2 \right)\\5\ldotp \;y=a+\textrm{bsin}\left(\textrm{cx}\right)-\textrm{c1}*\textrm{c2}\;\textrm{is}\;\textrm{not}\;\textrm{linear}\;\left(\;\textrm{coefficient}\;\textrm{complex}\;\textrm{is}\;\textrm{not}\;\textrm{linear}\right)\\6\ldotp 
% \;y=a+\textrm{bx}+x^c -a,b,{\textrm{can}}^{\prime } t\;\end{array}$$
% 
% _Note: Models with products of predictor variables may still be linear in 
% the fit coefficients. However, products of coefficients are not considered linear. 
% Also, taking the power of a predictor to the power of a coefficient will be 
% nonlinear._
% Course Example: Fuel Economy
% Load fuel economy data and view the relationships between the numeric predictors 
% and response.

load carEcon
carData
% 1.1. Fit a single predictor linear model of fuel economy
% Fit a single predictor linear model

figure
scatter(carTrain.EngDisp,carTrain.FuelEcon,'.')
mdl = fitlm(carTrain(:,{'EngDisp','FuelEcon'}))
%% 
% Predict fuel economy for the test data and compare the result with the actual 
% fuel economy of the test data.

econPred = predict(mdl,carTest.EngDisp);

figure
plot(carTest.EngDisp,econPred,'.')
hold on
scatter(carTest.EngDisp,carTest.FuelEcon,'.')
xlabel('Engine Displacement (L)')
ylabel('fuel Economy(L/100km) ')


%% 
% Plot distribution of errors

figure
err = carTest.FuelEcon - econPred;
histogram(err)
MSE = mean(err.^2)
xlabel('Prediction error')
ylabel('No. of economy fuel')
title(['MSE = ',num2str(MSE,4)])
%% 
% Plot ditribution of percentage errors

figure
percErr = 100 * err./carTest.FuelEcon;
histogram(percErr)
MAPE=mean(abs(percErr));
xlabel('Prediction percentage error')
ylabel('No. of economy fuel')
title(['MAPE = ',num2str(MAPE,4)])
% 1.2. Fit a two-predictor linear model of fuel economy
% NOTE!! When you use |fitlm|, use a table data type rather than an array data 
% type. Because it is much easier to use |'modelspec'| option with a table data 
% type.
% 
% For input format in |'modelspec'|, see _Wilkinson-Rogers_ notation.

doc wilkinson notation
%%
figure
scatter3(carTrain.EngDisp,carTrain.City_Highway,carTrain.FuelEcon,'.')
mdl = fitlm(carTrain(:,{'EngDisp','City_Highway','FuelEcon'}))
%% 
% Predict fuel economy for the test data and compare the result with the actual 
% fuel economy of the test data.

econPred = predict(mdl,carTest(:,{'EngDisp','City_Highway'}));

figure
subplot(2,2,1)
plot(carTest.FuelEcon,'.')
hold on
plot(econPred,'.')
title('Linear (dis + city/highway')
xlabel('Observations')
ylabel('Feul Economy')

subplot(2,2,2)
scatter(carTest.FuelEcon,econPred,'.')

xlimt = xlim;
hold on
plot(xlimt,xlimt,'k')
title('Linear (disp + city/highway')
xlabel('Actual Fuel economy')
ylabel('Predicted fuel economy')

subplot(2,2,3)
err = carTest.FuelEcon - econPred;
histogram(err)
MSE = mean(err.^2);
xlabel('Prediction eror')
title(['MSE = ',num2str(MSE,4)])

subplot(2,2,4)
percErr = 100 * err./carTest.FuelEcon;
histogram(percErr)
MAPE = mean(abs(percErr));
xlabel('Prediction percentage error')
title(['MAPE = ',num2str(MAPE,4)])


% 1.3. Fit a multivariate linear model of fuel economy

mdl = fitlm(carTrain);
econpred = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,econpred,'Linear (all predictor)')
%% 
% Specify the 'RobustOpts' option to reduce the influence of outliers

mdl = fitlm(carTrain,'RobustOpts',"cauchy");
econpred = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,econpred,'Linear (all predictor)')
%% 2. Stepwise Fitting
% 'Stepwise Fitting' is similar with 'Sequential Feature Selection' but provides 
% a simpler interface.
%% 
% * Sequential Feature Selection : Data + evaluation function
% * Stepwise fitting: Data + Criterion( sse(default), aic, bic, rsquared, adjrsquared 
% )
%% 
% Perform a stepwise linear fit of fuel economy

mdl = stepwiselm(carTrain);

%% 
% Predict fuel economy for the test data and compare the result with the actual 
% fuel economy of the test data.

econpred = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,econpred,'Stepwise')
%% 
% Change criterion and limit to linear models

mdl = stepwiselm(carTrain,'criterion','aic','Upper','interaction');
econpred = predict(mdl,carTest);
evaluatefit(carTest.FuelEcon,econpred,'Stepwise with aic')
%% 
% You can also use Sequential Feature Selection which provides a better result 
% but be careful of running this section because it takes long time.

%rng(1234)
%part = cvpartition(carData.FuelEcon,'KFold',10);
%[carDataNum, carDataNum_names] = cattable2mat(carData);
%% 
% Sequential feature selection on a Multiclass SVM.

%rng(1234)
%error = @(Xtrain , ytrain , Xtest , ytest)...
%    nnz(predict(fitcecoc(Xtrain , ytrain),Xtest)~= ytest);
%tokeep_nb = sequentialfs(error,carDataNum,carData.FuelEcon,'cv',part)

%SelectedPreditors = carDataNum(:,tokkep_nb);
%carDatNum_names(tokeep_nb);
%mdlSelected = fitcecoc(SelectedPreditors,carData.FuelEcon,'CVPartition',part);
%SelectedDataLoss = kfoldLoss(mdlSelected)
%% 
% Callback function: errorECOC function returns the number of inaccurate predictions 
% for |ytest| by a multiclass SVM classifier.

% function error = errorECOC(Xtrain,ytrain,Xtest,ytest)
% mdlnb = fitcecoc(Xtrain,ytrain);
% ypred = predict(mdlnb,Xtest);
% error = nnz(ypred ~= ytest);
% end