%% 2. Finding Natural Patterns in Data
%% Dimensionality Reduction
% When the dimension is large, it is difficult to visualize the data to look 
% for patterns or groups. You can try to visualize the data by transforming it 
% into a 2 or 3 dimenstional coordinate space.

load('bbstats.mat')
% 1. Multidimensional scaling

% open("2_1 Multidimensional Scaling.pdf") % Read this file before start
d = pdist(statsnorm)
[X,e ] = cmdscale(d)
pareto(e)
figure
scatter3(X(:,1),X(:,2),X(:,3))
view(110,40)

% 2. Principal Component Analysis

% open("2_2 Principal Component Analysis.pdf") % Read this file before start
[pcs,scrs,~,~,pexp] = pca(statsnorm)
pareto(pexp)
figure
scatter3(scrs(:,1),scrs(:,2),scrs(:,3))
view(110,40)
%% Unsupervised Learning
% 1. k-Means Clustering(1/3)

% open("2_3 k-Means Clustering.pdf") % Read this file before start
g = kmeans(statsnorm,2,"Replicates",5)
[g,c] = kmeans(statsnorm,2,"Replicates",5)
figure
scatter3(scrs(:,1),scrs(:,2),scrs(:,3),10,g)
view(110,40)
% 2. Gaussian Mixture Model(2/3)
% Grouping by GMM depends on parameter options. Try different 'RegularizationValue', 
% 'CovarianceType' options.

% open("2_4 GMM Clustering.pdf") % Read this file before start
%gm = fitgmdist(statsnorm,2,"Replicates",5,"RegularizationValue",0.02,"CovarianceType","diagonal")% Fit GMM to 'statsnorm' Data
 
% Use 2 instead of 1, because probability results are always helpful
g = cluster(gm,statsnorm) % 1. Cluster statsnorm according to the GMM and Output the grouping results
[g,~,p] = cluster(gm,statsnorm) % % 2. Cluster statsnorm according to the GMM and Output the grouping and probability results  

% Visualize groups
figure
scatter3(scrs(:,1),scrs(:,2),scrs(:,3),10,g)
view(110,40)

% Visualize probabilities
scatter3(scrs(:,1),scrs(:,2),scrs(:,3),10,p(:,1))
colorbar
view(110,40)
title("Probability of being in cluster 1")