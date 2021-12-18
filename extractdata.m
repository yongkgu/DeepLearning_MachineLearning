%% EXTRACTDATA
% Extract a subset of basketball data using indexing with tables.
%% Import data

stats = readtable('bball_stats.xlsx');
playerinfo = readtable('bball_players.xlsx');
%% Construct full names

firstname = playerinfo.useFirst;
%% 
% Replace any missing first names

idx = cellfun(@isempty,firstname);
firstname(idx) = playerinfo.firstName(idx);
%% 
% Join first and last name

fullname = strcat(firstname,{' '},playerinfo.lastName);
%% 
% Add back to the table

playerinfo.Name = fullname;
%% Extract needed player info

playerinfo = playerinfo(:,{'bioID','Name','pos','height','weight'});
%% Extract regular season statistics for 1990 onward
% Extract 1990 onward

idx = stats.year >= 1990;
stats = stats(idx,:);
%% 
% Throw away post-season stats

stats(:,24:end) = [];