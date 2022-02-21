import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer 

'''
Reads in the given data and returns relevant information into a dataframe

Take in a directory containing the files to read

returns a shuffled dataframe of the relevant information (CONTENT, CLASS)
'''

def read_data(file_dir):
        #print(file_dir)
        file_names = os.listdir(file_dir)
        os.chdir(file_dir)
        merged_df = pd.concat(map(pd.read_csv, file_names))
        merged_df.drop(['AUTHOR', 'DATE', 'COMMENT_ID'], axis = 1, inplace = True)
        return shuffle(merged_df)


def main():
        cwd = os.getcwd()
        comments_dir = cwd + '/comments'

        df = read_data(comments_dir)
        #Extracting Feature data
        comments = df['CONTENT'].to_list()
        classes = df['CLASS'].to_list()
        #Tokenizing and vectorizing comments 
        #comment_vectorizer = TfidfVectorizer(min_df = 1)
        comment_vectorizer = CountVectorizer()
        tokenized_comments = comment_vectorizer.fit_transform(comments)

        df = pd.DataFrame(tokenized_comments.todense(), columns = comment_vectorizer.get_feature_names_out())

        #Creating training and testing sets
        split_point = int(len(classes) * 0.8)

        x_train = df[:split_point]
        y_train = classes[:split_point]
        x_test = df[split_point:]
        y_test = classes[split_point:]

        #Fitting the Data
        nb = MultinomialNB()
        nb = nb.fit(x_train, y_train)

        #Scoring the model
        print(nb.score(x_test, y_test))

        #Most informative features for finding spam
        probs = np.array(nb.feature_log_prob_[1] - nb.feature_log_prob_[0])
        feature_probs = pd.DataFrame(probs, index = comment_vectorizer.get_feature_names_out(), columns=['Probability'])
        feature_probs.sort_values(['Probability'], ascending=False, inplace=True)

        #Top 10 more informative features
        print(feature_probs[:10])

        spam1 = "Hey, donate to my Eminem fund at this link, please!"
        spam2 = "Visit this website for a chance to go a Shakira concert for free!"
        ham1 = "Eminem is the greatest rapper alive. No doubt about it!"
        ham2 = "LMFAO really fell off the face of the earth but at least this song is still good."

        fake_spam = "Wow, I just checked out LMFAO's YouTube channel playlist. It's so cool that I just subscribed!"

        comments.extend([spam1, spam2, ham1, ham2, fake_spam])
        #print(comments[-5:])
        #new_vectorizer = TfidfVectorizer(min_df = 1)
        new_vectorizer = CountVectorizer()
        X = new_vectorizer.fit_transform(comments)
        df = pd.DataFrame(X.todense(), columns=new_vectorizer.get_feature_names_out()) 
        custom_comments = df[-5:]

        print (nb.predict(custom_comments))


if __name__ == "__main__":
        main()


    