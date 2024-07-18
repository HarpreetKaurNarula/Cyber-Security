//

content = fp.read()
                emails.append({'name': filename, 'content': content, 'category': category})
            except:
                print(f'skipped {filename}')
    return emails

ham = read_ham()
spam = read_spam()

df = pd.DataFrame.from_records(ham)
df = df.append(pd.DataFrame.froimport pandas as pd
import os

def read_spam():
    category = 'spam'
    directory = './enron1/spam'
    return read_category(category, directory)

def read_ham():
    category = 'ham'
    directory = './enron1/ham'

    import re

def preprocessor(e):
    return re.sub('[^A-Za-z]', ' ', e).lower()

    from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# The CountVectorizer converts a text sample into a vector (think of it as an array of floats).
# Each entry in the vector corresponds to a single word and the value is the number of times the word appeared.
# Instantiate a CountVectorizer. Make sure to include the preprocessor you previously wrote in the constructor.
vectorizer = CountVectorizer(preprocessor=preprocessor)

# Use train_test_split to split the dataset into a train dataset and a test dataset.
# The machine learning model learns from the train dataset.
# Then the trained model is tested on the test dataset to see if it actually learned anything.
# If it just memorized for example, then it would have a low accuracy on the test dataset and a high accuracy on the train dataset.
X_train,X_test,y_train,y_test = train_test_split(df["content"],df["category"],test_size=0.2,random_state=1)

# Use the vectorizer to transform the existing dataset into a form in which the model can learn from.
# Remember that simple machine learning models operate on numbers, which the CountVectorizer conveniently helped us do.
X_train_df = vectorizer.fit_transform(X_train)
