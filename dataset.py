from sklearn.datasets import load_iris
import pandas as pd 
import os 

# load iris data
iris = load_iris(as_frame=True)
df = pd.concat([iris.data , pd.Series(iris.target, name = 'target')], axis=1)

#create folder if not exists 
os.makedirs("data",exist_ok=True)

#save to csv
df.to_csv("data/iris.csv", index=False)
print("iris.csv has been created successfully.")