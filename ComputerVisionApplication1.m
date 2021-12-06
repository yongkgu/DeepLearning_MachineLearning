%% 7. Computer Vision Application - Object Detection(1/2)

load petGT.mat
% Peek into the data
% Investigate |'petGT'| table.

petGT           
%% 
% Read the first image in the table and display.

image1 = imread(petGT.imageFilename{1});
imshow(image1)
%% Inserting Annotation(Bounding box + Label) into Image (주석을 이미지 안에 넣기)
% The remaining variables in |'petGT'| table are the bounding boxes for each 
% pet. You can extract a bounding box using the same pattern as the filenames. 
% You should still use brackets({ }) to extract the bounding box.

annotLabel1 = "Madeline";
bbox1 = petGT.Madeline{1} % 바운딩 박스

%% 
% To view the image along with its label, you can add the bounding box to the 
% image using |insertObjectAnnotation| function

labeledIm1 = insertObjectAnnotation(image1,"rectangle",bbox1,annotLabel1)
imshow(labeledIm1)

%% Preprocessing Ground Truth
% For object detection, every image has a labeled bounding box. If you preprocess 
% your images, you need to be careful to update the bounding boxes accordingly.
% 
% Many object detectors don't require images to be the same size, but scaling 
% your training data to a consistent size can help avoid issues during training. 
% Since the bounding boxes are stored with pixel locations, you need to scale 
% the bounding boxes if you scale your images.
%     Create an image datastore and a box label datastore

imDS = imageDatastore(petGT.imageFilename);
boxDS = boxLabelDatastore(petGT(:,2:end))

%     Combine datastores 

combinedDS = combine(imDS,boxDS);
%     Resize images and bounding boxes

resizedDS = transform(combinedDS,@scaleGT)
%     Preview the resized data
% To view a preprocessed ground truth, you can preview the first image in the 
% datastore

newGT = preview(resizedDS)
resizedData = insertObjectAnnotation(newGT{1},"rectangle",newGT{2},newGT{3});
imshow(resizedData)

% Support function
% |scaleGT| resizes images to |targetSize|. It also uses the same scale to resize 
% the corresponding bounding boxes.

function data = scaleGT(data)  
    targetSize = [224 224];
    % data{1} is the image
    scale = targetSize./size(data{1},[1 2]);
    data{1} = imresize(data{1},targetSize);
    % data{2} is the bounding box
    data{2} = bboxresize(data{2},scale);
end