%% Improving Predictive Models
%% 1. Cross Validation
% Use 10-fold cross validation to evaluate the various predictive models for 
% the heart disease data.

% Run comparison with holdout validation
compareclassifiers
%%
% Add variable to hold cross validation errors
results.kFoldLoss = zeros(height(results),1);

% Remove pruned trees because the cross validation cannot apply for the pruned trees
results([5 7],:) = []
%%
% Make a cross validation partition
rng(1234)
part = cvpartition(heartdataNum.HeartDisease,'kFold',10)

% 1.1. Create cross-validated models
% Note that the cross-validated models are only used to evaluate performance. 
% The fitted models do not have a |predict| method. 

% Divide the data into training+validation and test sets
rng(1234)
part = cvpartition(heartdataNum.HeartDisease,"HoldOut",0.2);
trValidx = training(part);
trValData = heartdataNum(trValidx,:)% trValData is to build a model
testData = heartdataNum(~trValidx,:)  % Assume that testData is newly measured data.

% Divide the training+validation data into training and validation sets
part2 = cvpartition(trValData.HeartDisease,"HoldOut",0.2);
tridx = training(part2);
trdata = trValData(tridx,:);
valdata = trValData(~tridx,:);

% Models with 'holdout' do have a predict method for the validation data
mdlknn = fitcknn(trdata,"HeartDisease");
errKNN_tr = resubLoss(mdlknn)
errKNN_val = loss(mdlknn,valdata)

HDpred = predict(mdlknn,valdata);
HDtrue = valdata.HeartDisease;
figure
confusionchart(HDtrue,HDpred)

% Models with 'holdout' do have a predict method for the test data
HDpred_test = predict(mdlknn,testData)

% k-fold
part2 = cvpartition(trValData.HeartDisease,"KFold",10);
mdlknn_kfold = fitcknn(trValData,'HeartDisease','CVPartition',part2)
errKNN_kfold = kfoldLoss(mdlknn_kfold)

% Models with 'kfold' do NOT have a predict method
%HDpred_test_kfold = predict(mdlknn_kfold,testData) % k-fold에서는 predict할 수 있는 기능이 없다 
%% 2. Hyperparameter optimization - 모델 내부 설정을 하는 파라미터인데 하이퍼파라미터는 파라미터 중의 하나로써 일단 학습이 시작하면 변하지 않는 값들이다. 대표적 파라미터가 weight인데 따로 설정하는 것이 아니고 일반적으로 그 파라미터들은 데이터로부터 얻어지는 값들이기 때문에 학습이 진행되면 될수록 값이 변한다. 그것을 파라미터라고 하는데 그 중에서 학습 하기 전에 우리가 직접 입력해서 학습 중간에 값이 변하지 않는 파라미터를 하이퍼 파라미터라 한다, 
% Hyperparameter optimization is very powerful in that it automatically tune 
% a model. Just do it.

% Divide the data into training and validation sets
load heartdisease.mat
rng(1234)
%evalclusters 비지도 학습 
part = cvpartition(heartdataNum.HeartDisease,'HoldOut',0.2);
tridx = training(part);
trData = heartdataNum(tridx,:);
valData = heartdataNum(~tridx,:);

% Train a kNN model
mdlknn = fitcknn(trData,'HeartDisease');
errKNN_tr = resubLoss(mdlknn)
errKNN_val = loss(mdlknn,valData)



% 2.1. Turn on hyperparameter optimization - 가장 시간이 오래 걸려
% Setting '|OptimizeHyperparamters|' to '|auto|' will optimize these _k_-NN 
% properties:
%% 
% * |NumNeighbors|
% * |Distance|

%mdlknn_HO = fitcknn(trdata,'HeartDisease','OptimizeHyperparameters',"auto");
%% 
% The command-line output contains information on the evaluated property values 
% and the hyperparameters chosen for the final model.

mdlknn_HO.Distance 
mdlknn_HO.NumNeighbors
%% 
% The first plot shows the best objective function against the iteration number. 
% The second plot is a model of the objective function against the parameters. 
% You can hide these plots with the '|ShowPlots|' argument.

errKNNHO_tr = resubLoss(mdlknn_HO)
errKNNHO_val = loss(mdlknn_HO,valdata)
% 2.2. Change optimization options
% Make cross validation partition

part = cvpartition(trData.HeartDisease,"KFold",10);
%% 
% Change the validation to use '|part|' and decrease the number of objective 
% evaluations.

opt = struct('CVPartition',part,'MaxObjectiveEvaluation',20);
%% 
% Setting '|OptimizeHyperparamters|' to '|all|' will optimize these _k_-NN properties:
%% 
% * |NumNeighbors|
% * |Distance|
% * |DistanceWeight|
% * |Exponent|
% * |Standardize|

%mdlknn_HO = fitcknn(trData,'HeartDisease','OptimizeHyperparameters','all','HyperparameterOptimizationOptions',opt);
%% 
% Note that there is only one plot displayed. You will only see the second plot 
% if there are one or two optimization parameters, and if '|Optimizer|' is set 
% to '|bayesopt|'.

errKNN_tr = resubLoss(mdlknn_HO)
errKNN_val = loss(mdlknn_HO,valData)