%% HIERARCHICALCLUSTERS
% Make and interpret clusters from the dendrogram.
%% Load data

load bbstats
idx = data.pos == 'G';
guardstats = zscore(stats(idx,:));
%% Perform hierarchical clustering

Z = linkage(guardstats,'ward');
dendrogram(Z)
%% Make and interpret clusters from the dendrogram

gc2 = cluster(Z,'maxclust',2);
gc3 = cluster(Z,'maxclust',3);
%% 
% Two clusters

figure
parallelcoords(guardstats,'Group',gc2,'Quantile',0.25)
%% 
% Add overall average

hold on
parallelcoords(guardstats,'Quantile',0.5,'Color','k')
labelXTicks(labels);
hold off
%% 
% Three clusters

figure
parallelcoords(guardstats,'Group',gc3,'Quantile',0.25)
%% 
% Add overall average

hold on
parallelcoords(guardstats,'Quantile',0.5,'Color','k')
labelXTicks(labels);
hold off
%% Dig deeper into the differences between groups
% It appears that the primary dividing factor is the amount of time a player 
% plays per game (which correlates strongly to their stats in general). One way 
% to understand this difference in game time might be to look at per-minute stats.
% 
% Divide stats by minutes per game 

permin = bsxfun(@rdivide,guardstats(:,4:end),guardstats(:,3));
%% 
% Look at per-minute stats by cluster

figure
parallelcoords(permin,'Group',gc2,'Quantile',0.25)
labelXTicks(labels(4:end));
figure
parallelcoords(permin,'Group',gc3,'Quantile',0.25)
labelXTicks(labels(4:end));
%% 
% There is a slight stratification by points-scoring stats (points/field goals/free 
% throws/three-pointers). But it's not directly correlated to game time. The clusters 
% in gc3 are:
% 
% 1 = low game time, average points-scoring
% 
% 2 = average game time, below-average points-scoring
% 
% 3 = high game time, above-average points-scoring
%% Interpret group splits further down the dendrogram
% Split into seven clusters

figure
dendrogram(Z,'ColorThreshold',18)
gc7 = cluster(Z,'MaxClust',7);
%% 
% The middle group corresponds to #2 in gc3. See how it divides

figure
parallelcoords(guardstats(gc3==2,:),'Group',gc7(gc3==2),'Quantile',0.25)
labelXTicks(labels);
%% 
% Note that the main differentiator now is size (height and weight). Also, the 
% smaller players have slightly higher points-scoring stats.