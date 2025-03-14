import numpy as np
import pandas as pd
from collections import defaultdict
import re

class NaiveBayes():
    """
    This is a Naive Bayes spam filter that learns word spam probabilities 
    from pre-labeled training data and predicts whether emails are ham or spam.
    """
    def __init__(self):
        """
        Initialize variables that will be used for training and prediction.
        """
        self.num_train_hams = 0
        self.num_train_spams = 0
        self.word_counts_spam = {}
        self.word_counts_ham = {}
        self.HAM_LABEL = 'ham'
        self.SPAM_LABEL = 'spam'
        
    def get_word_set(self, email_text: str) -> set:
        """
        Extracts a set of unique words from an email text.
        
        :param email_text: The text content of an email
        :return: A set of all unique words in the email
        """
        # Clean the text and split into words
        text = email_text.replace('\r', '').replace('\n', ' ')
        words = text.split(' ')
        return set(words)
    
    # TODO implement:
    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Trains the Naive Bayes model on the provided DataFrame.
        
        :param train_df: DataFrame with 'email' and 'label' columns
        :return: None
        """
        # 1. Count number of ham and spam emails
        counts = train_df['label'].value_counts()
        self.num_train_hams = counts[self.HAM_LABEL]
        self.num_train_spams = counts[self.SPAM_LABEL]

        # 2. Reinitialize word count dictionaries
        self.word_counts_spam = defaultdict(lambda : 1)
        self.word_counts_ham = defaultdict(lambda : 1)

        # 3. Process each email in the training set

        self.sum_words_ham = 0
        self.sum_words_spam = 0
        for i, row in train_df.iterrows():
            words = self.get_word_set(row['email'])

            if row['label'] == self.HAM_LABEL:
                target = self.word_counts_ham  
            else:
                target = self.word_counts_spam
            counter = 0 
                
            for word in words:
                if len(word) < 2:
                    continue
                counter += 1
                target[word] += 1

            if row['label'] == self.HAM_LABEL:
                self.sum_words_ham += counter
            else:
                self.sum_words_spam += counter
        
                
    # TODO implement:
    def predict(self, email_text: str) -> str:
        """
        Predicts whether a single email is ham or spam.
        
        :param email_text: The text content of an email
        :return: The predicted label ('ham' or 'spam').
        """
        # 1. Get words in the email
        words = self.get_word_set(email_text)

        # 2. Calculate prior probabilities
        spam_prob = self.num_train_spams / (self.num_train_hams + self.num_train_spams)
        ham_prob = self.num_train_hams / (self.num_train_hams + self.num_train_spams)
        
        # 3. Calculate log probabilities for ham and spam (for computational reasons)
        log_prob_spam = np.log(spam_prob)
        log_prob_ham = np.log(ham_prob)

        # 4. For each word in the email, update the log probabilities
        for word in words:
            if len(word) < 2:
                continue
            log_prob_spam += np.log(self.word_counts_spam[word] / self.num_train_spams)
            log_prob_ham += np.log(self.word_counts_ham[word] / self.num_train_hams)
        

        # 5. Predict label (in case of a tie, return HAM_LABEL)
        if log_prob_ham >= log_prob_spam:
            return self.HAM_LABEL
        return self.SPAM_LABEL
    
    
    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        """
        Predicts ham/spam labels for all emails in a DataFrame.
        
        :param df: DataFrame with 'email' column
        :return: Series with predicted labels
        """
        predictions = []
        for _, row in df.iterrows():
            predictions.append(self.predict(row['email']))
        return pd.Series(predictions)
        
    def accuracy(self, test_df: pd.DataFrame) -> float:
        """
        Computes the accuracy of predictions on a test DataFrame.
        
        :param test_df: DataFrame with 'email' and 'label' columns
        :return: Accuracy as a float between 0 and 1
        """
        correct = 0
        total = len(test_df)
        
        for index, row in test_df.iterrows():
            prediction = self.predict(row['email'])
            if prediction == row['label']:
                correct += 1
                
        return correct / total
    
    
class MultinomialBayes(NaiveBayes):
    def get_word_set(self, email_text: str) -> set:
        # Clean the text and split into words
        text = email_text.replace('\r', '').replace('\n', ' ')
        words = text.split(' ')
        return words
    
    def predict(self, email_text: str) -> str:

        # 1. Get words in the email
        words = self.get_word_set(email_text)

        # 2. Calculate prior probabilities
        spam_prob = self.num_train_spams / (self.num_train_hams + self.num_train_spams)
        ham_prob = self.num_train_hams / (self.num_train_hams + self.num_train_spams)
        
        # 3. Calculate log probabilities for ham and spam (for computational reasons)
        log_prob_spam = np.log(spam_prob)
        log_prob_ham = np.log(ham_prob)

        # 4. For each word in the email, update the log probabilities
        for word in words:
            if len(word) < 2:
                continue
            log_prob_spam += np.log(self.word_counts_spam[word] / self.sum_words_spam)
            log_prob_ham += np.log(self.word_counts_ham[word] / self.sum_words_ham)
        

        # 5. Predict label (in case of a tie, return HAM_LABEL)
        if log_prob_ham >= log_prob_spam:
            return self.HAM_LABEL
        return self.SPAM_LABEL
