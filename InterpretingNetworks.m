%% 2. Interpreting Network - Convolution, ReLU, Max Pooling Layers
%% Activations

image = imread("face.jpg")
sz = size(image)
imshow(image)
net = squeezenet; % Load SqueezeNet
layers = net.Layers % Save and view the layers of SqueezeNet
%% 
% View the network architecture

analyzeNetwork(net)
% Show Activations of First Convolutional Layer
% Activation: Information passed between the network layers.
% 
% You can extract the activations from a layer by using |activations| function.

layers(2) % View the hyperparameters and learnable parameters of the first convolution layer
activn1 = activations(net,image,"conv1"); % Extract the activations from the first convolution layer

% Display the activation extracted from "conv1"
montage(activn1)
montage(mat2gray(activn1))
% Investigate the activations in specific channels
% Each tile in the grid of activations is the output of a channel in the |conv1| 
% layer. White pixels represent strong positive activations and black pixels represent 
% strong negative activations. A channel that is mostly gray does not activate 
% as strongly on the input image. The position of a pixel in the activation of 
% a channel corresponds to the same position in the original image. A white pixel 
% at some location in a channel indicates that the channel is strongly activated 
% at that position.
% 
% For channel 41,

actvn1_ch41 = activn1(:,:,41);
actvn1_ch41 = mat2gray(actvn1_ch41);

actvn1_ch41 = imresize(actvn1_ch41,sz(1:2)) % resize the activation in channel 41 in order to compare it with the original image
% imshow, montage
montage({image,actvn1_ch41}) % Display the original image and the activation in channel 41

%% 
% For channel 22,

actvn1_ch22 = activn1(:,:,22);
actvn1_ch22 = mat2gray(actvn1_ch22);

actvn1_ch22 = imresize(actvn1_ch22,sz(1:2)) % resize the activation in channel 22 in order to compare it with the original image
montage({image,actvn1_ch22}) % Display the original image and the activation in channel 22

%% Deeper Layer Activations
% Most convolutional neural networks learn to detect features like color and 
% edges in their first convolutional layer. In deeper convolutional layers, the 
% network learns to detect more complicated features. Later layers build up their 
% features by combining features of earlier layers. Investigate the |fire6-squeeze1x1| 
% layer in the same way as the |conv1| layer. Calculate, reshape, and show the 
% activations in a grid.

actvn6 = activations(net,image,"fire6-squeeze1x1") % Extract the activations from "fire6-squeeze1x1" layer

% Display the activation extracted from "fire6-squeeze1x1"
montage(mat2gray(actvn6))
%% 
% For channel 47,

actvn6_ch47 = actvn6(:,:,47);
actvn6_ch47 = mat2gray(actvn6_ch47);
actvn6_ch47 = imresize(actvn6_ch47,sz(1:2))  % resize the activation in channel 41 in order to compare it with the original image

montage({image,actvn6_ch47}) % Display the original image and the activation in channel 41
%% Following Activations in the Network Architecture
% Rectified Linear Unit (ReLU) Layer
% ReLu layer apply a threshold to an input such that every value less than zero 
% is set to zero. You will see that any negative activations from a convolution 
% layer will be set to zero after passing through the ReLU layer.

actvn6_relu = activations(net,image,"fire6-relu_squeeze1x1") % Extract the activations from "fire6-relu_squeeze1x1" layer
actvn6_relu_ch47 = actvn6_relu(:,:,47);
actvn6_relu_ch47 = mat2gray(actvn6_relu_ch47);
actvn6_relu_ch47 = imresize(actvn6_relu_ch47,sz(1:2));

montage({image,actvn6_ch47,actvn6_relu_ch47},"size",[1 3])



% Max Pooling Layer
% Max pooling layers perform downsampling, which causes large activations to 
% be more pronounced after passing through the pooling layer.

cnnLayerBehavior = imread("cnnLayerBehavior.jpg");
imshow(cnnLayerBehavior)