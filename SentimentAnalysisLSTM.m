%%reference to deeplearning using LSTM classifier implementation 
%https://uk.mathworks.com/help/textanalytics/ug/classify-text-data-using-deep-learning.html?searchHighlight=Create%20Simple%20Text%20Model%20for%20Classification&s_tid=srchtitle_Create%20Simple%20Text%20Model%20for%20Classification_6
%importing data and assigning it to a variable
%clear variables and console
clc; clear;
filename = "reviews_12302.csv";
% filename = "amazon_cells_labelled.txt";

% filename = "imdb_labelled_2.txt";
data = readtable(filename,'TextType','string');
head(data)
% reading the contents of reviewFile in a string table
data.result = categorical(data.result);

%categorizing the different categories of the result
% columns


% creating visual reprsentation of the categoris
% on a bar chart
figure
histogram(data.result);
xlabel("Class")
ylabel("Frequency")
title("Class Distribution")



%dividing the data to training and testing data
%holding 10% of the data for testing
cvp = cvpartition(data.result,'Holdout',0.1);
dataTrain = data(training(cvp),:);
dataValidation = data(test(cvp),:);

%assigning the training and tesing  review 
% column to trainData and testData and the
% results column to YTrain and YTest
textDataTrain = dataTrain.review;
textDataValidation = dataValidation.review;
YTrain = dataTrain.result;
YValidation = dataValidation.result;

%checking if the correct data is imported
%using a word cloud
figure
wordcloud(textDataTrain);
title("Training Data")


%preprocessing the data using the function
%the function preprocessText
documentsTrain = preprocessText(textDataTrain);
documentsValidation = preprocessText(textDataValidation);

%assign training columns
documentsTrain(1:2);

%LSTM network only works with sequences 
% therefore, convert document to sequences 
% using word encoding function

enc = wordEncoding(documentsTrain);

documentLengths = doclength(documentsTrain);
figure
histogram(documentLengths)
title("Document Lengths")
xlabel("Length")
ylabel("Number of Documents")

% make all the document sequences of the 
%same length, in this case 25
sequenceLength = 45;
XTrain = doc2sequence(enc,documentsTrain,'Length',sequenceLength);
XTrain(1:2)

%convertint the testing data using same option
XValidation = doc2sequence(enc,documentsValidation,'Length',sequenceLength);

%createa LSTM Layers
% create and include a sequence input layer and 
% include an LSTM layer and set 
% the number of hidden units to 80. 
% To use the LSTM layer for a sequence-to-label
% classification, set the output mode to
% 'last'. Finally, add a fully connected 
% layer with the same size as 
% the number of classes, 
% a softmax layer, and a classification layer.

inputSize = 1;
embeddingDimension = 50;
numHiddenUnits = 120;

numWords = enc.NumWords;
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numWords)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

%%Specify the training options:

% Train using the Adam solver.
% Specify a mini-batch size of 16.
% Shuffle the data every epoch.
% Monitor the training progress by setting the 'Plots' option to 'training-progress'.
% Specify the validation data using the 'ValidationData' option.
% Suppress verbose output by setting the 'Verbose' option to false.

options = trainingOptions('adam', ...
    'MiniBatchSize',16, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation, YValidation}, ...
    'Plots','training-progress', ...
    'Verbose',false);

% Train the LSTM network using the 
% trainNetwork function.
net = trainNetwork(XTrain,YTrain,layers,options);


% Classify the event type of three new reports.
% Create a string array containing the new reports.
testArray = [ ...
    "With its gorgeous cinematography...lovely muted score, and sure-footed performances, especially by Frances O'Connor, Patricia Rozema's Mansfield Park sets a new standard for adaptations of Jane Austen's work."
    "Mansfield Park is a witty, entertaining film, and I hope I haven't made it sound too serious."
    "The story is simple -- almost painfully so -- but it still could have worked had the screenplay been more focused and innovative, and the characters dealt with more satisfyingly."
    "Little more than a warmed-over TV skit."];


% Preprocess the text data using the 
% preprocessing steps as the training 
% documents.

documentsNew = preprocessText(testArray);
% Convert the text data to sequences
% using doc2sequence with the same options as when creating the training sequences.

XTest = doc2sequence(enc,documentsNew,'Length',sequenceLength);
% Classify the new sequences using the trained
% LSTM network.
%this should return Leak, ElectronicFailure and Mechanical failure
labelsNew = classify(net,XTest)
