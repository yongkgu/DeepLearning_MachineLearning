%% FINDBESTK
% Try to determine the best number of clusters for the basketball statistics.
%% Load data

load bbstats
%% Make silhouette plots for different numbers of clusters

for k = 1:4
    subplot(2,2,k)
    rng(1234)
    gm = fitgmdist(statsnorm,k+1,'Replicates',5,'RegularizationValue',0.02,'Options',statset('MaxIter',900));
    g = cluster(gm,statsnorm);
    silhouette(statsnorm,g)
end
%% Automate with EVALCLUSTERS

figure
%% 
% Use silhouette value

rng(1234)
ce = evalclusters(statsnorm,'gmdistribution','silhouette','KList',2:8)
subplot(2,1,1)
plot(ce.InspectedK,ce.CriterionValues)
xlabel('Number of clusters')
ylabel('Silhouette value')
%% 
% Use Calinski-Harabasz criterion value

rng(1234)
ce = evalclusters(statsnorm,'gmdistribution','CalinskiHarabasz','KList',2:8)
subplot(2,1,2)
plot(ce.InspectedK,ce.CriterionValues)
xlabel('Number of clusters')
ylabel('Calinski-Harabasz criterion value')