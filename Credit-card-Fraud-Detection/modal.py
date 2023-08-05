import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix


class Credit_Card_Modal:
    def load_data(self):
        df = pd.read_csv('creditcard.csv')
        self.preprocessing(df)

    def preprocessing(self, data):
        df = pd.DataFrame(data)
        df = df.drop_duplicates()
        self.create_Graph(df)
        self.create_Modal(df)

    def create_Graph(self, data):
        # corrmat = data.corr()
        # fig = plt.figure(figsize=(12, 9))
        # sns.heatmap(corrmat, vmax=.8, square=True)
        # plt.show()
        # dividing the X and the Y from the dataset
        pass

    def create_Modal(self, data):
        X = data.drop(['Class'], axis=1)
        Y = data["Class"]
        print(X.shape)
        print(Y.shape)
        xData = X.values
        yData = Y.values
        xTrain, xTest, yTrain, yTest = train_test_split(
            xData, yData, test_size=0.2, random_state=42)
        rfc = RandomForestClassifier()
        rfc.fit(xTrain, yTrain)
        # predictions
        yPred = rfc.predict(xTest)
        print("The model used is Random Forest classifier")

        acc = accuracy_score(yTest, yPred)
        print("The accuracy is {}".format(acc))

        prec = precision_score(yTest, yPred)
        print("The precision is {}".format(prec))

        rec = recall_score(yTest, yPred)
        print("The recall is {}".format(rec))

        f1 = f1_score(yTest, yPred)
        print("The F1-Score is {}".format(f1))

        MCC = matthews_corrcoef(yTest, yPred)
        print("The Matthews correlation coefficient is{}".format(MCC))
        LABELS = ['Normal', 'Fraud']

        conf_matrix = confusion_matrix(yTest, yPred)
        plt.figure(figsize=(12, 12))
        sns.heatmap(conf_matrix, xticklabels=LABELS,
                    yticklabels=LABELS, annot=True, fmt="d")
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.show()


if __name__ == "__main__":
    cf = Credit_Card_Modal()
    cf.load_data()
