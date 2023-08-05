import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
class Credit_Card_Modal:
    def load_data(self):
        df = pd.read_csv('creditcard.csv')
        self.preprocessing(df)
    def preprocessing(self,data):
        print(data.head())



if __name__=="__main__":
    cf = Credit_Card_Modal()
    cf.load_data()