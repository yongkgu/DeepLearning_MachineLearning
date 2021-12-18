%% 2. Finding Natural Patterns in Data
%% Dimensionality Reduction
% When the dimension is large, it is difficult to visualize the data to look 
% for patterns or groups. You can try to visualize the data by transforming it 
% into a 2 or 3 dimenstional coordinate space.

load('bbstats.mat')
% 1. Multidimensional scaling

% open("2_1 Multidimensional Scaling.pdf") % Read this file before start
d= pdist(statsnorm)
[X,e] = cmdscale(d)
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
g = kmeans(statsnorm,2)
[g,c] = kmeans(statsnorm,2,"Replicates",5)
figure
scatter3(scrs(:,1),scrs(:,2),scrs(:,3),10,g)
view(110,40)
% 2. Gaussian Mixture Model(2/3)
% Grouping by GMM depends on parameter options. Try different 'RegularizationValue', 
% 'CovarianceType' options.

% open("2_4 GMM Clustering.pdf") % Read this file before start
gm = fitgmdist(statsnorm,2,"Replicates",5,"RegularizationValue",0.02,"CovarianceType","diagonal")% Fit GMM to 'statsnorm' Data
 
% Use 2 instead of 1, because probability results are always helpful
g = cluster(gm,statsnorm)% 1. Cluster statsnorm according to the GMM and Output the grouping results
[g,~,p] = cluster(gm,statsnorm) % 2. Cluster statsnorm according to the GMM and Output the grouping and probability results  

% Visualize groups
figure
scatter3(scrs(:,1),scrs(:,2),scrs(:,3),10,g)
view(110,40)
% Visualize probabilities
figure
scatter3(scrs(:,1),scrs(:,2),scrs(:,3),10,p(:,1))
colorbar
view(110,40)
title("Probability of being cluster1")
% 3. Interpreting the Groups
% 1. Visualizing observations in cluseters grouped by k-Means

[g, c ] = kmeans(statsnorm,2,"Replicates",5)
%중요 
parallelcoords(statsnorm,"Group",g)%'parallelcoords' function to visualize observation in clussters
xticklabels(labels)% Is it discrete visualization?
parallelcoords(statsnorm,"Group",g,"Quantile",0.25)% Much better now
parallelcoords(c,"Group",1:2)% Use the centroid to visualize
%% 
% 2. Correlation between the groups and player positions (IMPORTANT because 
% you can visualize clustering quality)

% counts = crosstab(yCat,grp)
%% 
% 

positions = data.pos
counts1 = crosstab(data.pos,g)

figure
bar(counts1,"stacked")
xticklabels(positions)
legend({'1','2'})

counts2 = crosstab(g,data.pos)
figure
bar(counts2,"stacked")
xticklabels({'1','2'})
legend(positions)
% 4. Evaluating Clustering Quality

% Group data into 2~5 clusters and see their silhouette values.
% Is it easy to find the optimal number of clusters?
[g,c] = kmeans(statsnorm,2,"Replicates",5)
silhouette(statsnorm,g)
[g,c] = kmeans(statsnorm,3,"Replicates",5)
silhouette(statsnorm,g)
[g,c] = kmeans(statsnorm,4,"Replicates",5)
silhouette(statsnorm,g)
[g,c] = kmeans(statsnorm,5,"Replicates",5)
silhouette(statsnorm,g)
% cluster를 몇 개의 그룹으로 나누어야 할지 결정 'evalclusters' function finds you the optimal number of clusters
clustev1 = evalclusters(statsnorm,"kmeans","silhouette","KList",2:5)
clustev2 = evalclusters(statsnorm,"kmeans","CalinskiHarabasz","KList",2:5)
clustev3 = evalclusters(statsnorm,"kmeans","DaviesBouldin","KList",2:5)
clustev4 = evalclusters(statsnorm,"kmeans","gap","KList",2:5)
%  5. Hierarchical Clustering(3/3)

% Extract the statistics only for player listed as guards
idx = data.pos =='G'
numGuard = sum(idx)

% guardstats = statsnorm(idx,:)
% Try to use 'zscore' function this time
guardstats = zscore(stats(idx,:))

% Make a dendrogram of the guards' statistics
Z = linkage(guardstats,"ward")
dendrogram(Z)

% Make and interpret clusters from the dendrogram
gc2 = cluster(Z,"maxclust",2)
gc3 = cluster(Z,"maxclust",2)

figure
parallelcoords(guardstats,"Group",gc2,"Quantile",0.25)
parallelcoords(guardstats,"Group",gc3,"Quantile",0,25)

% 'evalclusters' function with 'linkage' option
% also find you the optimal number of clusters
clustEv1 = evalclusters(guardstats,"linkage","silhouette","KList",2:5)
clustEv2 = evalclusters(guardstats,"linkage","CalinskiHarabasz","KList",2:5)
clustEv3 = evalclusters(guardstats,"linkage","DaviesBouldin","KList",2:5)
clustEv4 = evalclusters(guardstats,"linkage","gap","KList",2:5)