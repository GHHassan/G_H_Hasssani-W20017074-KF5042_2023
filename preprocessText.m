function documents = preprocessText(textData)
%nonempty = any(~ismissing(textData), "all");

% Create a new table without empty rows
%documents = textData(nonempty,:);

% Tokenize the text.
documents = tokenizedDocument(textData);
documents = removeLongWords(documents, 15);
%Erase tags
%documents = eraseTags(documents);
% Convert to lowercase.
documents = lower(documents);

% Erase punctuation.
documents = erasePunctuation(documents);

%Erase stop words

end