%% CLUSTERVAR
% Use hierarchical clustering to cluster the basketball statistics variables.
%% Load data

load bbstats
%% Find linkage between variables
% Transpose the |statsnorm| vector to find linkage between the basketball statistics 
% variables. 

varnorm = statsnorm';
Z = linkage(varnorm,'ward');
%% Create dendrogram

a=axes;
[~,~,outperm] = dendrogram(Z);
xlabel('Variables')
ylabel('Distance')
ticklabels = labels(outperm);
a.XTickLabel = ticklabels;
a.XTickLabelRotation = 90;
%% Cluster the linkage into 4 clusters

grp = cluster(Z,'maxclust',4);
%% 
% View the clusters assigned to each variable in a table.

table(labels',grp)
%% View the clusters by color

a=axes;
[~,~,outperm] = dendrogram(Z,'ColorThreshold',45);
xlabel('Variables')
ylabel('Distance')
ticklabels = labels(outperm);
a.XTickLabel = ticklabels;
a.XTickLabelRotation = 90;