function outputLabel = FaceRecognition2(trainPath, testPath)
%% VJ-LBP
%  Face detection: Viola Jones Algorithm
%  Feature extraction: Local Binary Patterns

%% Retrieve training images and labels
folderNames = ls(trainPath);
traininLabels = folderNames(3:end,:);
data = uint8(zeros(600, 600, length(traininLabels)));

for ii = 1:length(traininLabels)
    path = fullfile(trainPath, traininLabels(ii,:));
    filenames = dir(path);
    filenames = sort({filenames.name});
    fullpath = fullfile(path, filenames{3}); 
    image = imread(fullpath);
    image = rgb2gray(image);
    data(:,:,ii) = image;
end

%% Face and eyes detection using Viola Jones algorithm
%  Eyes are detected to straighten the image to make the face images pose invariant
processedData = uint8(zeros(100, 100, length(traininLabels)));
faceDetector = vision.CascadeObjectDetector();
RightEyeDetector = vision.CascadeObjectDetector('RightEyeCART');
LeftEyeDetector = vision.CascadeObjectDetector('LeftEyeCART');

for i = 1:length(traininLabels)
    im = data(:,:,i);
    
    % Spliting the image in half
    n = fix(size(im,2)/2);
    left = im(:,1:n,:);
    right = im(:,n+1:end,:);
    % Detect eyes
    bboxr= step(RightEyeDetector,left);
    bboxl= step(LeftEyeDetector,right);
    
    if isempty(bboxr) == 1 || isempty(bboxl) == 1 || size(bboxr, 1) > 1 || size(bboxl, 1) > 1
        % do nothing if eyes are not detected due to occlusion
    else
        bboxl(1)=bboxl(1)+n;
        % rotate image to straighten it
        im = imrotate(im,(180/pi)*atan((bboxr(2)-bboxl(2))/(bboxr(1)-bboxl(1))));
    end
    
    bbox = step(faceDetector, im);
    if isempty(bbox) == 1 || size(bbox, 1) > 1
        % if not a front facing face just use the same image
        im = imresize(im, [100 100]);
    else
        % else crop around the detected face
        im = imcrop(im, bbox);
        im = imresize(im, [100 100]);
    end
    
    processedData(:,:,i) = im;
end

%% Feature extraction of all training images using LBP

featureMapsTrain = zeros(length(traininLabels), 23600);
for i = 1:length(traininLabels)
    featureMap = extractLBPFeatures(processedData(:,:,i), ...
                'NumNeighbors',8, ...
                'Radius',1, ...
                'Interpolation','Linear', ...
                'CellSize',[5 5]);
            
    featureMapsTrain(i,:) = featureMap;
end

featureMapsTrain = featureMapsTrain.';

%% Test set
outputLabel=[];
testImgNames = dir(testPath);
testImgNames = testImgNames(3:end,:);
testImgNames = sort({testImgNames.name});

for i = 1:length(testImgNames)
    fullpath = fullfile(testPath, testImgNames{i});
    image = imread(fullpath);
    
    % Carry out same pre-processing as training images
    image = rgb2gray(image);
    n = fix(size(image,2)/2);
    left = image(:,1:n,:);
    right = image(:,n+1:end,:);
    bboxr= step(RightEyeDetector,left);
    bboxl= step(LeftEyeDetector,right);
    
    if isempty(bboxr) == 1 || isempty(bboxl) == 1 || size(bboxr, 1) > 1 || size(bboxl, 1) > 1
        % do nothing
    else
        bboxl(1)=bboxl(1)+n;
        image = imrotate(image,(180/pi)*atan((bboxr(2)-bboxl(2))/(bboxr(1)-bboxl(1))));
    end
    
    bbox = step(faceDetector, image);
    if isempty(bbox) == 1 || size(bbox, 1) > 1
        image = imresize(image, [100 100]);
    else
        image = imcrop(image, bbox);
        image = imresize(image, [100 100]);
    end
    
    % Extract features
    featureMap = extractLBPFeatures(image, ...
                    'NumNeighbors',8, ...
                    'Radius',1, ...
                    'Interpolation','Linear', ...
                    'CellSize',[5 5]);

    % Make prediction
    featureMap = featureMap.';
    ccValue=featureMapsTrain'*featureMap;
    labelIndx=find(ccValue==max(ccValue));
    outputLabel=[outputLabel;traininLabels(labelIndx(1),:)]; 
end   

end

