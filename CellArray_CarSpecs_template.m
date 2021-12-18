clear; clc;
CarSpecs = readtable('CarSpecs.xls')
%plot(CarSpecs.EngineSize, CarSpecs.Power)
scatter(CarSpecs.EngineSize, CarSpecs.Power)
xlabel("Engine Size(L)")
ylabel("Power [hp]")
title("Power vs. Engine Size")
%%
% Use './' for element divide cal. within matrices
CarSpecs.PtW = CarSpecs.Power./CarSpecs.Weight

% Max function 
%[MaxPtW, idx]=max(CarSpecs.PtW)
[MaxPtW, idx] = max(CarSpecs.PtW)
[MinPtw, idx5] = min(CarSpecs.PtW)
%% 4. Cell array
% Each cell can contain any type of data as you can see in 'CarSpecs' variable

% Comparison CarSpecs.Model(idx), CarSpecs.Model{idx}
CarSpecs.Model
CarSpecs.Model(idx) %cell array 형태로 나타남 
CarSpecs.Model{(idx)} % use curly brace '{}' to see content in cell

CarSpecs.Model
CarSpecs.Model(idx5)
CarSpecs.Model{(idx5)}

% Sort function
[byPtW , idx1] = sortrows(CarSpecs,"PtW","descend")
[byPtW2, idx6] = sortrows(CarSpecs,"PtW","ascend")
% Compare these two results : they are same
byPtW(1:5,[1,2,end])
byPtW(1:5, {'Make','Model','PtW'})