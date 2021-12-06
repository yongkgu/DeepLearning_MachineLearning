%% 8. Classifying Sequence Data - Dummify Variables
%% Convert Categorical to Numerical by Creating Dummy Variables
%     1. character --> uint8
% Save the phrase 'to be or not to be' to a variable

t = 'to be or not to be'
%% 
% Convert the characters to a numeric array using |unit8| function.

t_num = uint8(t)
%     2. uint8 --> categorical
% Convert the numeric array to a categorical array using |categorical| function.

t_cat = categorical(t_num)
categories(t_cat)
%     3. categorical --> dummy variable
% Create dummy variables using |dummyvar| function. If the categorical array 
% is represented as a row vector, you need to transpose it because |dummyvar| 
% function expects a column vector. 

t_dum = dummyvar(t_cat')' % 행백터를 받는게 아니라 열백터를 받기 때문에
%% Create dummy variables for the whole lower case letters

vocab = uint8(' abcdefghijklmnopqrstuvwxyz')
t_catAll = categorical(t_num,vocab)  % c=vocab으로 카테고리컬 하는데 t_num을 데이터로 쓰겠다.
categories(t_catAll)
t_dumAll = dummyvar(t_catAll')'