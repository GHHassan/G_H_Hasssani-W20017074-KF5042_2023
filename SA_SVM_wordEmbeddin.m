% clearing the work space from variables
% clearing the command window
clear;
clc; 
% extracting positive words from the specified folder
% folder contains files of pre-labelled english words
% extracting the positive and Negative words 
% to a cell Char Array (P and N)
% assigning the Positive and Negative P and N
% to String Arrays wordsPositive and wordsNegative
% finally, closing the file
findPositive = fopen(fullfile('opinion-lexicon-English','positive-words.txt'));
P = textscan(findPositive, '%s', 'CommentStyle', ';');
findNegative = fopen(fullfile("opinion-lexicon-English","negative-words.txt"));
N = textscan(findNegative, '%s', 'CommentStyle', ';');
wordsNegative = string(N{1});
wordsPositive = string(P{1});
fclose all;

%%define a hashTable
% this is to make the data retrieval fast
% hashTable has an O(1) retrieval speed
words_Hash = java.util.Hashtable;

%%Put all the positive and negative words in the hash, 
% giving them a value of 1 and -1 respectively.
for ii = 1:size(wordsPositive)
    words_Hash.put(wordsPositive(ii,1),1);
end
for jj = 1:size(wordsNegative)
    words_Hash.put(wordsNegative(jj,1),-1)
end

%% word-based sentiment analyses =============================

% filename = "imdb_labelled_2.txt";
% filename = "yelp_labelled.txt";
filename = "reviews_12302.csv";
%filename = "amazon_cells_labelled.txt";
dataReviews = readtable(filename,'TextType','string'); 
textData = dataReviews.review; %get review text 
actualScore = dataReviews.result; %get review sentiment

sents = preprocessReviews(textData);

sentimentScore = zeros(size(sents));
for ii = 1 : sents.length
    docwords = sents(ii).Vocabulary;
    for jj = 1 : length(docwords)
        if words_Hash.containsKey(docwords(jj))
            sentimentScore(ii) = sentimentScore(ii) + words_Hash.get(docwords(jj));
        end
    end
end

% importing pretrained word embedding
emb = fastTextWordEmbedding;
words = [wordsPositive;wordsNegative]; 
labels = categorical(nan(numel(words),1)); 
labels(1:numel(wordsPositive)) = "Positive";
labels(numel(wordsPositive)+1:end) = "Negative";
% removing the words that dont exist on the vocabulary
data = table(words,labels,'VariableNames',{'Word','Label'});
idx=~isVocabularyWord(emb,data.Word);
data(idx,:) = [];

% dividing the data in training and testing parts
% using cvp function. set aside the 10% of words for testing
numwords = size(data,1);
cvp = cvpartition(numwords, 'HoldOut', 0.01);
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp), :);
% training a lexicon based model using
% support Vector Machine (SVM) pretrained
% word embedding
wordsTrain = dataTrain.Word;
Xtrain = word2vec(emb, wordsTrain);
Ytrain = dataTrain.Label;
model = fitcsvm(Xtrain,Ytrain);

%%test classifiers
wordsTest = dataTest.Word;
XTest = word2vec(emb,wordsTest);
YTest = dataTest.Label;

%prediect the sentiment lables of the test word vectors
[Ypred, scores] = predict(model,XTest);

%visualise the classification accuracy in a confussion matrix
figure
confusionchart(YTest,Ypred, 'ColumnSummary','column-normalized');
sentimentScore(sentimentScore > 0) = 1;   %take > 1 to be 1 only
sentimentScore(sentimentScore < 0)= -1;   %there is no neutral only negative

 notfound = sum(sentimentScore == 0);
 covered = numel(sentimentScore)- notfound;
 tp=0; tn=0; count=0;
 for i=1:length(actualScore)
     if sentimentScore(i)==1 && actualScore(i)=="Fresh"
         tp=tp+1; count=count+1;
     elseif sentimentScore(i)==-1 && actualScore(i)== "Rotten"
         tn=tn+1; count=count+1;
     end
 end
 accuracy = (tp+tn)*100/covered; 
 coverage=covered*100/numel(sentimentScore);
 
 disp(coverage)
 disp(accuracy)
