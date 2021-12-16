from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns

#We evalute the classifier, simple
def evaluate(true, pred):
    precision = precision_score(true, pred, average= 'macro')
    recall = recall_score(true, pred, average = 'macro')
    f1 = f1_score(true, pred, average = 'macro')
    cf = confusion_matrix(true, pred)
    cf = sns.heatmap(cf, annot=True)
    print(cf)
    print(f"ACCURACY SCORE:\n{accuracy_score(true, pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n\tPrecision: {precision:.4f}\n\tRecall: {recall:.4f}\n\tF1_Score: {f1:.4f}")
    
    
#Fucntion to get the scores and add them to a liste of scores for final comparaison
def get_score(true, pred, classifier_name):
    
    precision = precision_score(true, pred, average= 'macro')
    recall = recall_score(true, pred, average = 'macro')
    f1 = f1_score(true, pred, average = 'macro')
    accuracy = accuracy_score(true, pred)
    cf = confusion_matrix(true, pred)
    cf = sns.heatmap(cf, annot=True)
    a, b = [classifier_name, precision, recall, f1, accuracy], cf
    
    return a, b