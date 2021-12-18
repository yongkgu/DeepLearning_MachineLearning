clear; clc;
stats = readtable('bball_stats.xlsx');
playerinfo = readtable('bball_players.xlsx');
%% Construct full names

firstname = playerinfo.useFirst% Extract useFirst
idx = ismissing(firstname) % Indexing missing firstnam
firstname(idx) = playerinfo.useFirst(idx) % Replace missing data with 'firstName' data
 % Check missing in firstname variable
%%
 fullname = strcat(firstname,{' '},playerinfo.useFirst)% Join first and last name with a space in between
 playerinfo.Name = fullname% Add fullname to the last column of the table
%% Extract needed player info

height_weight_table = playerinfo(:,{'height','weight'})% Extract height & weight in table
height_weight_array = playerinfo{:,{'height','weight'}} % EXtract height & weight in array
%%
% Extract playerinfo in table
playerinfo = playerinfo(:,{'bioID','Name','pos','height','weight'})
%playerinfo = playerinfo{:,{'bioID','Name','pos','height','weight'}} % This line doesn't work because their types are cell and double.
%% Extract regular season statistics for 1990 onward

% Extract 1990 onward
idx = stats >= 1990
stats = stats(idx,:)

% Throw away post-season stats
stats(:,(24:end)) = []
%% Categorical data

%positions = {'G','G-F','F-G','F','F-C','C-F','C'} 
 positions = {'G','G-F','F-G','F','F-C','C-F','C'} %create 'positions' variable
 playerinfo.pos = categorical(playerinfo.pos)% Convert playerinfo.pos to a categorical array 
 categories(playerinfo.pos)% How many categories?
 playerinfo.pos = categorical(playerinfo.pos,positions)% Force to convert playerinfo.pos to 'positions' categorical array
 categories(playerinfo.pos)% Now how many categories? It should be seven 
%% Grouped operations

stats = stats(:,[1 6 8:end])
totalstats = grpstats(stats,'playerID','sum')%Total basketball statistics for each player 중요 grpstats
idx = totalstats.sum_GP<100 %Remove players who have played fewer than 100 games
totalstats(idx,:) = []
totalstats.GroupCount = [];
%% Table properties - 주의깊게 보기 

vars = stats.Properties.VariableNames; % Extract the variable names from the table of statistics
totalstats.Properties.VariableNames = vars % Replace the variable names with 'vars' variable
%% Merging data - 굉장히 중요 교집합으로 합치기 innerjoin

 %data = innerjoin(playerinfo,totalstats,"LeftKeys","bioID","RightKeys","playerID");
 data = innerjoin(playerinfo,totalstats,"LeftKeys","bioID","RightKeys","playerID")% % doc innerjoin
%% Working with missing data

sum(ismissing(data))% How many missing observations for each variable?
% 1. First way to remove the observations with missing data
idx = isundefined(data.pos)% Find observations with undefined 'data.pos'
 % remove the observations with undefined 'data.pos'
data(idx , :) = [];
% 2. Second way to remove the observations with missing data
%data=rmmissing(data)

%% Explore data

%boxplot(data.height,data.pos)
gscatter(data.rebounds,data.points,data.pos) % See the result. Is it meaningful? If not why?
gscatter(data.rebounds./data.GP, data.points./data.GP , data.pos) % Divide by game played. Still linearly proportional
gscatter(data.rebounds./data.minutes, data.points./data.minutes , data.pos)% Divide by minutes played. Now data seems meaningful.
%% Normalizing data - 정규화하는것. 

stats = data{:, 7:end};
statsnorm = bsxfun(@rdivide, stats , data.GP);
data{:, 7:end} = statsnorm;