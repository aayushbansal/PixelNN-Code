%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Author: Aron Yu
% Updated: 09.22.2016
%
% Purpose: Demonstrates the correct experimental setup using provided data.
%          Extract a single train/test split and perform learning + evaluation.
%
% Attribute Index: 1 = open, 2 = pointy, 3 = sporty, 4 = comfort
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

format short g;

%% Ordered Pairs (UT-Zap50K-1)
%  Iteration through all 10 splits for each attribute to compute the mean 
%  accuracy for comparison.

clear all; close all; clc;

load zappos-gist; load zappos-color;
load zappos-labels;
load train-test-splits;

% Determine Attribute & Split
attrIndex = 1;
splitIndex = 1;

imageFeats = [gistfeats colorfeats];   % combine GIST + Color feats
imagePairs = mturkOrder{attrIndex};    % extract pair w/ labels

% Global Train/Test Pair Indices
trainIndex = trainIndexAll{attrIndex}{splitIndex};
testIndex = testIndexAll{attrIndex}{splitIndex};

% Image Pairs w/ Labels
trainPairs = imagePairs(trainIndex,:);
testPairs = imagePairs(testIndex,:);

% Relative Labels (1: A > B, 2: B < A)
trainLabels = trainPairs(:,4);
testLabels = testPairs(:,4);

% Features
trainFeatsA = imageFeats(trainPairs(:,1),:);   % image A
trainFeatsB = imageFeats(trainPairs(:,2),:);   % image B

testFeatsA = imageFeats(testPairs(:,1),:);     % image A
testFeatsB = imageFeats(testPairs(:,2),:);     % image B

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INSERT YOUR TRAINING + PREDICTION CODE HERE   %
%                                               %
% Train On: trainFeatsA trainFeatsB trainLabels %
% Test On: testFeatsA testFeatsB                %
% Results: predLabels => 1: A > B, 2: B < A     %
%                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Evaluate Performance
resultRawOrder = (testLabels == predLabels);
resultAccOrder = mean(resultRawOrder);   % prediction accuracy


%% Fine-Grained Pairs (UT-Zap50K-2)
%  Train on ordered pairs and test on fine-grained pairs. Only 1 iteration.

clear all; close all; clc;

load zappos-gist; load zappos-color;
load zappos-labels; load zappos-labels-fg;

% Determine Attribute
attrIndex = 1;

imageFeats = [gistfeats colorfeats];   % combine GIST + Color feats
trainPairs = mturkOrder{attrIndex};
testPairs = mturkHard{attrIndex};

% Relative Labels (1: A > B, 2: B < A)
trainLabels = trainPairs(:,4);
testLabels = testPairs(:,4);

% Features
trainFeatsA = imageFeats(trainPairs(:,1),:);   % image A
trainFeatsB = imageFeats(trainPairs(:,2),:);   % image B

testFeatsA = imageFeats(testPairs(:,1),:);     % image A
testFeatsB = imageFeats(testPairs(:,2),:);     % image B

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INSERT YOUR TRAINING + PREDICTION CODE HERE   %
%                                               %
% Train On: trainFeatsA trainFeatsB trainLabels %
% Test On: testFeatsA testFeatsB                %
% Results: predLabels => 1: A > B, 2: B < A     %
%                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Evaluate Performance
resultRawHard = (testLabels == predLabels);
resultAccHard = mean(resultRawHard);   % prediction accuracy

