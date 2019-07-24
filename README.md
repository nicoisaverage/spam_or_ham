# Naive Bayes Classifier w/ Improvements

Improvements to simple Naive Bayes classifier built to detect spam or ham in the Enron spam corpus. Wrapper for kyotocabinet.db was written by @coleifer. 

Base Classifier Accuracy: 85.33%

Improvements to be made to classifier accuracy:

1. Eliminate common stopwords
2. Check for words with capital letters 
3. Treat words in email like individual features and look for patterns in spam messages
4. Treat bigrams as features and look for spam specific bigrams
5. Record all caps or links in emails as boolean features 

Modified Classifier Accuracy: 
