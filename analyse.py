import argparse
import json
import pandas as pd
import numpy as np
from sklearn import metrics

def main():
    parser = argparse.ArgumentParser(description="Experiment preparator and run")
    parser.add_argument("config", type=str, help="path to json defining the experiment")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = json.load(file)
        y = pd.read_csv(config["Experiment"]["TestDataLabels"])
        result = pd.read_csv(config["Experiment"]["ResultFile"])
        id = config["Experiment"]["IdName"]
        target = config["Experiment"]["TargetValueName"]
        d1 = {}
        d2 = {}
        for index, object in y.iterrows():
            #print(index, object)
            d1[int(object[id])] = object[target]
        for index, object in result.iterrows():
            d2[int(object[id])] = object[target]
        #print(result)
        ys = []
        predicted = []
        for key in d1:
            ys.append(d1[key])
            predicted.append(d2[key])
        predicted = np.array(predicted).astype(float)
        ys = np.array(ys).astype(int)
        #print (predicted[0:10], ys[0:10])
        fpr, tpr, thresholds = metrics.roc_curve(ys, predicted, pos_label=1)
        print(metrics.auc(fpr, tpr))


if __name__ == "__main__":
    main()