import pandas as pd

#Importing the data

url="https://raw.githubusercontent.com/TADY25/Unil_Tissot/main/data/training_data.csv"
data = pd.read_csv(url)
data.drop(columns = ['id'], inplace = True)