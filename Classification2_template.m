%%  3. Building Classification Models
%% Supervised Learning
% Course Example: Heart Disease

load('heartdisease.mat')

[g,c] = kmeans(zscore(heartdataNum{:,1:11}),2)
counts = crosstab(g,heartdataNum.HeartDisease);
figure
bar(counts,'stacked')
legend([{'true'},{'false'}])
% Training and Validation

% Divide the heart disease into training(80%) and validation(20%) sets
rng(1234)
part = cvpartition(heartdataNum.HeartDisease,'HoldOut',0.2);
tridx = training(part);
tdata = heartdataNum(tridx,:);
vdata = heartdataNum(~tridx,:);
tdataAll = heartdataAll(tridx,:);
vdataAll = heartdataAll(~tridx,:);

% 1. Nearest Neighbor Classification(1/2)
% The most straightforward way of classification without any assumptions about 
% the underlying distribution of the data

mdl = fitcknn(tdata,'HeartDisease');
% Why use tdata, not tdataAll? because k-NN does not accept mixed predictors(numeric+categorical).
% Evaluating Classifications

mdl = fitcknn(tdata,'HeartDisease')
errKNN_train = resubLoss(mdl)
errKNN_test = loss(mdl,vdata)

%Visualize prediction accuracy
HDpred = predict(mdl,vdata);
HDtrue = vdata.HeartDisease;
figure
confusionchart(HDtrue,HDpred)

% 1. Nearest Neighbor Classification(2/2)

mdl = fitcknn(tdata,'HeartDisease','NumNeighbors',5)
errKNN_train = resubLoss(mdl)
errKNN_test = loss(mdl,vdata)
HDpred = predict(mdl,vdata);
HDtrue = vdata.HeartDisease;
figure
confusionchart(HDtrue,HDpred)
% 2. Decision Tree Classification
% A decision tree model make a categorical response prediction by following 
% a sequence of binary decisions based on predictor variable value. It does not 
% assume any underlying distributions of the data like k-NN.
% 
% A decision tree model is robust to noise data.

mdl = fitctree(tdata,'HeartDisease')
view(mdl,'Mode','graph')
errTree_train = resubLoss(mdl)
errTree_test = loss(mdl,vdata)

mdl = prune(mdl,'Level',3)
view(mdl,'Mode',"graph")
errTree_train = resubLoss(mdl)
errTree_test = loss(mdl,vdata)

mdl = fitctree(tdataAll,'HeartDisease')
errTree_train = resubLoss(mdl)
errTree_test = loss(mdl,vdataAll)

mdl = prune(mdl,'Level',3);
errTree_train = resubLoss(mdl)
errTree_test = loss(mdl,vdataAll)
% 3. Naive Bayes Classification
% A Naive Bayes model uses Bayes's rule of conditional probability to estimate 
% the probability of a given observation being in each response class. Accordingly, 
% it assumes an underlying distribution of the data. (Note: Each predictor variable 
% should be independent, If not, use a 'Discriminant Analysis')

mdl = fitcnb(tdata,'HeartDisease')
errNB_train = resubLoss(mdl)
errNB_test = loss(mdl,vdata)

dists = [repmat({'kernel'},1,11) repmat({'mvmn'},1,10)]
mdl = fitcnb(tdata,'HeartDisease','DistributionNames','kernel');
errNB_train = resubLoss(mdl)
errNB_test = loss(mdl,vdata)

mdl = fitcnb(tdataAll,'HeartDisease');
errNB_train = resubLoss(mdl)
errNB_test = loss(mdl,vdataAll)

mdl = fitcnb(tdataAll,'HeartDisease','DistributionNames',dists);
errNB_train = resubLoss(mdl)
errNB_test = loss(mdl,vdataAll)
% 4. Discriminant Analysis - 각각의 predicter들이 독립적이든 독립적이지 않든 신경 쓰지 않느다.  
% Similar to Naive Bayes, discriminant analysis works by assuming that the observations 
% in each prediction class can be modeled with a normal probability distribution. 
% However, there is no assumption of independence in each predictor.

mdl = fitcdiscr(tdata,'HeartDisease')
errDA_train = resubLoss(mdl)
errDA_test = loss(mdl,vdata)

mdl = fitcdiscr(tdata,'HeartDisease','DiscrimType',"quadratic")
errDA_train = resubLoss(mdl)
errDA_test = loss(mdl,vdata)
% 5. Support Vector Machines - 전반적으로 모든 경우에 있어서 항상 상위권의 분류기 정확도 - reponse가 2개 인 경우  
% A SVM model classifies data by finding the best hyperplane that separates 
% all data points
% 
% It only works for binary categories (true/false, 0/1, pass/fail)

mdl = fitcsvm(tdata,'HeartDisease')
errSVM_train = resubLoss(mdl)
errSVM_test = loss(mdl,vdata)

mdl = fitcsvm(tdata,'HeartDisease','KernelFunction',"gaussian");
errSVM_train = resubLoss(mdl)
errSVM_test = loss(mdl,vdata)
% 5. Multiclass Support Vector Machines - 기본 svm 자체로는 reponse variable이 2개일 경우에만 사용 가능 하지만 svm을 사용할 일은 거의 없을것. msvm도 2개에서 사용 가능 
% Multiclass SVM by creating an error-correcting output codes (ECOC).

mdl = fitcecoc(tdata,'HeartDisease');
errMSVM_train = resubLoss(mdl)
errMSVM_TEST = loss(mdl,vdata)

t = templateSVM('KernelFunction','gaussian')
mdl = fitcecoc(tdata,'HeartDisease','Learners',t)
errGaussianMSVM_train = resubLoss(mdl)
errGuussianMSVM_test = loss(mdl,vdata)