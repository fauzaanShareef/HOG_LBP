function  outputLabel=Benchmark(trainPath, testPath)
%%   A simple face reconition method using cross-correlation based tmplate matching.
%    trainPath - directory that contains the given training face images
%    testPath  - directory that constains the test face images
%    outputLabel - predicted face label for all tested images 

%% Retrieve training images and labels
folderNames=ls(trainPath);
trainImgSet=zeros(600,600,3,length(folderNames)-2); 
labelImgSet=folderNames(3:end,:); 
for i=3:length(folderNames)
    imgName=ls([trainPath, folderNames(i,:),'\*.jpg']);
    trainImgSet(:,:,:,i-2)= imread([trainPath, folderNames(i,:), '\', imgName]);
end

%% Prepare the training image: Here we simply use the gray-scale values as template matching. 

trainTmpSet=zeros(600*600,size(trainImgSet,4)); 
for i=1:size(trainImgSet,4)
    tmpI= rgb2gray(uint8(trainImgSet(:,:,:,i)));
    tmpI=double(tmpI(:))/255'; 
    trainTmpSet(:,i)=(tmpI-mean(tmpI))/std(tmpI); 
end

%% Face recognition for the test images
testImgNames=ls([testPath, '*.jpg']);
outputLabel=[];
for i=1:size(testImgNames,1)
    testImg=imread([testPath, testImgNames(i,:)]);
    %perform the same pre-processing as the training images
    tmpI= rgb2gray(uint8(testImg));
    tmpI=double(tmpI(:))/255';               
    tmpI=(tmpI-mean(tmpI))/std(tmpI); 
    ccValue=trainTmpSet'*tmpI;               
    labelIndx=find(ccValue==max(ccValue));   
    outputLabel=[outputLabel;labelImgSet(labelIndx(1),:)];   
end
