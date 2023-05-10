function [documents] = preprocessReviews(textData)
    % to lower case
    cleanTextData = lower(textData);
    % tokenizaion
    documents = tokenizedDocument(cleanTextData);
    % erase punctuations
    documents = erasePunctuation(documents);
    % removing stop words that doesnt bear any sentiment
    documents = removeWords(documents,stopWords);
end


