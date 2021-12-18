%% DIMREDUCTION
% Compare PCA and multidimensional scaling.
%% Import data

load bbstats
posnames = {'G','G-F','F-G','F','F-C','C-F','C'};
position = data.pos;
%% Perform multidimensional scaling

d = pdist(statsnorm);
[X,e] = cmdscale(d);
figure
pareto(e)
figure
scatter3(X(:,1),X(:,2),X(:,3))
view(110,40)
%% Perform PCA

[pcs,scrs,~,~,pexp] = pca(statsnorm);
figure
pareto(pexp)
figure
scatter3(scrs(:,1),scrs(:,2),scrs(:,3))
view(110,40)
%% Compare PCA and CMD scaling
% Note that CMD scaling is the same as PCA when using the 2-norm as the distance 
% metric (within a potential minus sign). In this case, it turns out that the 
% 3rd component is flipped:

figure
scatter3(X(:,1),X(:,2),-X(:,3))
view(110,40)
%% Look for correlation with player position

scatter3(scrs(:,1),scrs(:,2),scrs(:,3),10,position)
view(110,40)
c = colorbar;
c.TickLabels = posnames;