import pandas as pd

#Importing the data

def get_train_data():
    url="https://raw.githubusercontent.com/TADY25/Unil_Tissot/main/data/training_data.csv"
    data = pd.read_csv(url)
    data.drop(columns = ['id'], inplace = True)
    
    return data

def get_unlabelled_data():
    url="https://raw.githubusercontent.com/TADY25/Unil_Tissot/main/data/unlabelled_test_data.csv"
    unlabelled = pd.read_csv(url)
    unlabelled.drop(columns = ['id'], inplace = True)
    
    return unlabelled

def get_sub_data():
    url="https://raw.githubusercontent.com/TADY25/Unil_Tissot/main/data/sample_submission.csv"
    sub = pd.read_csv(url)
    sub.drop(columns = ['id'], inplace = True)
    
    return sub

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns

def split_df(dataframe):
    corpus = dataframe['sentence'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(corpus, df['difficulty'].tolist(),
                                                        test_size = 0.2,
                                                        random_state = 0)
    return X_train, X_test, y_train, y_test