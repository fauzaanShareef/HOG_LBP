clear all;
close all;
trainPath=''; % provide full path here
testPath='';
%% Baseline Method
tic;
   outputLabel=Benchmark(trainPath, testPath);
baseLineTime=toc

load testLabel
correctP=0;
for i=1:size(testLabel,1)
    if strcmp(outputLabel(i,:),testLabel(i,:))
        correctP=correctP+1;
    end
end
recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy%

%% VJ-HOG
tic;
   outputLabel1=FaceRecognition1(trainPath, testPath);
method1Time=toc

load testLabel
correctP=0;
for i=1:size(testLabel,1)
    if strcmp(outputLabel1(i,:),testLabel(i,:))
        correctP=correctP+1;
    end
end

recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy%


%% VJ-LBP
tic;
   outputLabel2=FaceRecognition2(trainPath, testPath);
method2Time=toc

load testLabel
correctP=0;
for i=1:size(testLabel,1)
    if strcmp(outputLabel2(i,:),testLabel(i,:))
        correctP=correctP+1;
    end
end
recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy%
