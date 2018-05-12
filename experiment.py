import argparse
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import realpath, dirname, join

def main():
    parser = argparse.ArgumentParser(description="Experiment preparator and run")
    parser.add_argument("config", type=str, help="path to json defining the experiment")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = json.load(file)
        prepare_boosting()
        df = pd.read_csv(config["Experiment"]["Data"])
        change_targets(df, config["Experiment"]["TargetValueName"])
        y = df[:][ [config["Experiment"]["IdName"], config["Experiment"]["TargetValueName"]] ]
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
        X_test.drop(config["Experiment"]["TargetValueName"], axis = 1)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        train = realpath(join(dirname(config["Experiment"]["Data"]), "sliced_train.csv"))
        test = realpath(join(dirname(config["Experiment"]["Data"]), "sliced_test.csv"))
        test_y = realpath(join(dirname(config["Experiment"]["Data"]), "sliced_test_y.csv"))
        config_path = realpath(args.config)
        print (test, config_path)
        config["Experiment"]["TrainData"] = train
        config["Experiment"]["TestData"] = test
        config["Experiment"]["TestDataLabels"] = test_y
        os.system("rm " + test_y)
        os.system("rm " + test)
        os.system("rm " + train)
        X_train.to_csv(train, header=True, index=None, sep=',', float_format='%.3f')
        X_test.to_csv(test, header=True, index=None, sep=',', float_format='%.3f')
        y_test.to_csv(test_y, header=True, index=None, sep=',', mode='a')


    with open(args.config, "w") as file:
        json.dump(config, file, sort_keys=True, indent = 4)

    os.system("./FantasticGradientBoosting " + args.config)


def prepare_boosting():
    os.system("cmake ./CMakeLists.txt")
    os.system("make")


def change_targets(df, name):
    idx = 0
    names = {}
    for index, object in df.iterrows():
        el = object[name]
        if el not in names:
            names[el] = idx
            idx += 1
        df.at[index, name] = names[el]


if __name__ == "__main__":
    main()