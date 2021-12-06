%% 0. Spectrogram for Image Classification
%% Read Audio Data

[y, fs] = audioread("pianoSound.m4a") % [시간에 따른 데이터 , 샘플링 프리퀀시]
 
%%
sound(y(:,1),fs)
%% Comparison Between Plot and Spectrogram

plot(y(:,1))
pspectrum(y(:,1),fs,"spectrogram")
ylim([0,5])

%% Remove the Unnecessary
% Remove axis, colorbar and title for image classification

hold on
% 표 이미지 화
axis off % 축 삭제
colorbar off
ttitle("")
hold off