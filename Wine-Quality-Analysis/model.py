import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier


class WineQuality:
    def load_data(self):
        data = pd.read_csv('WineQT.csv')
        df = pd.DataFrame(data)
        self.preprocessing(df)
    def preprocessing(self,df):
        df = df.drop('Id',axis=1)
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())
        print(df.nunique())
        df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
        df = df.drop('quality',axis=1)
        features = df.drop(['best quality'], axis=1)
        target = df['best quality']
        xtrain, xtest, ytrain, ytest = train_test_split(
            features, target, test_size=0.2, random_state=40)
        models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

        for i in range(3):
            models[i].fit(xtrain, ytrain)

            print(f'{models[i]} : ')
            print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
            print('Validation Accuracy : ', metrics.roc_auc_score(
                ytest, models[i].predict(xtest)))
            print()

        metrics.plot_confusion_matrix(models[1], xtest, ytest)
        plt.show()


if __name__ == "__main__":
    wq = WineQuality()
    wq.load_data()