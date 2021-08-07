function outputLabel = FaceRecognition1(trainPath, testPath)
%% VJ-HOG
% Face detection: Viola Jones Algorithm
% Feature extraction: HOG

%% Retrieve training images and labels
folderNames = ls(trainPath);
traininLabels = folderNames(3:end,:);

data = uint8(zeros(600, 600, length(traininLabels)));
for i = 1:length(traininLabels)
    path = fullfile(trainPath, traininLabels(i,:));
    filenames = dir(path);
    filenames = sort({filenames.name});
    fullpath = fullfile(path, filenames{3}); 
    image = imread(fullpath);
    image = rgb2gray(image);
    data(:,:,i) = image;
end

%% Face detection using Viola Jones algorithm
processedData = zeros(100, 100, length(traininLabels));
faceDetector = vision.CascadeObjectDetector();

for i = 1:length(traininLabels)
    im = data(:,:,i);    
    bbox = step(faceDetector, im);
    
    if isempty(bbox) == 1 || size(bbox, 1) > 1
        % if not a front facing face just resize
        im = imresize(im, [100 100]);
    else
        % else crop around the detected face and resize
        im = imcrop(im, bbox);
        im = imresize(im, [100 100]);
    end
    im = imresize(im, [100 100]);
    im = im2double(im);
    processedData(:,:,i) = im;
end

%% Feature extraction using HOG
featureMapsTrain = zeros(length(traininLabels), 38988);

for i = 1:length(traininLabels)
    features = extractHOGFeatures(processedData(:,:,i), ...
        'CellSize',[5 5], ...
        'BlockSize',[2 2], ...
        'NumBins', 27);
    featureMapsTrain(i,:) = features;
end

featureMapsTrain = featureMapsTrain.';

%% Making predictions for test data
outputLabel=[];
testImgNames = dir(testPath);
testImgNames = testImgNames(3:end,:);
testImgNames = sort({testImgNames.name});

for i = 1:length(testImgNames)
    fullpath = fullfile(testPath, testImgNames{i});
    image = imread(fullpath);
    
    % Carry out same pre-processing as training images
    image = rgb2gray(image);
    bbox = step(faceDetector, image);
    
    if isempty(bbox) == 1 || size(bbox, 1) > 1
        % if not a front facing face just resize
        image = imresize(image, [100 100]);
    else
        % else crop around the detected face and resize
        image = imcrop(image, bbox);
        image = imresize(image, [100 100]);
    end
   
    image = im2double(image);
    
    % Extract features
    features = extractHOGFeatures(image, ...
        'CellSize',[5 5], ...
        'BlockSize',[2 2], ...
        'NumBins', 27);
    
    % Make prediction
    features = features.';
    ccValue=featureMapsTrain'*features;
    labelIndx=find(ccValue==max(ccValue));
    outputLabel=[outputLabel;traininLabels(labelIndx(1),:)]; 
end   

end

