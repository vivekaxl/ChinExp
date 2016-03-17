from __future__ import division
from os import listdir
from os.path import isfile, join

import model

class data_item():
    def __init__(self, id, decisions, objective):
        self.id = id
        self.decisions = decisions
        self.objective = objective

    def __repr__(self):
        return str(self.id)+ "|" +",".join(map(str, self.decisions)) + "|" + str(self.objective)


def read_csv(filename, header=False):
    def transform(filename):
        return "./Raw_Data/" + filename

    import csv
    data = []
    with open(transform(filename), 'r') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            if i == 0 and header is False: continue  # Header
            elif i ==0 and header is True:
                H = row
                continue
            data.append(data_item(i, map(float, row[:-1]), float(row[-1])))
    if header is True: return H, data
    return data


def transform(filename):
    return "./Raw_Data/" + filename


def where_data_transformation(filename):
    from Utilities.WHERE.where import where
    # The Raw_Data has to be access using this attribute table._rows.cells
    import pandas as pd
    df = pd.read_csv(filename)
    headers = [h for h in df.columns if '$<' not in h]
    data = df[headers]
    clusters = where(data)

    return clusters


def model_cart(training_indep, training_dep, testing_indep, testing_dep):
    from sklearn import tree
    CART = tree.DecisionTreeRegressor()
    CART = CART.fit(training_indep, training_dep)

    predictions = [float(x) for x in CART.predict(testing_indep)]
    mre = []
    for i, j in zip(testing_dep, predictions):
        if i != 0:
            mre.append(abs(i - j) / float(i))  # abs(original - predicted)/original
        else:
            if i==j: mre.append(0)

    return mre


def experiment(filename):
    content = read_csv(filename)
    len_data = len(content)
    content_dict = {}
    for c in content:
        key = ",".join(map(str, c.decisions))
        content_dict[key] = float(c.objective)

    clusters = where_data_transformation(transform(filename))

    train_independent = []
    train_dependent = []

    test_independent = []
    test_dependent = []
    for cluster in clusters:
        indexes = list(range(len(cluster)))
        from random import choice
        random_point_index = choice(indexes)
        train_independent.append(cluster[random_point_index])

        key = ",".join(map(str, map(float, cluster[random_point_index])))
        train_dependent.append(content_dict[key])

        # remove the training sample from test set
        del indexes[random_point_index]

        for index in indexes:
            test_independent.append(cluster[index])
            key = ",".join(map(str, map(float, cluster[index])))
            test_dependent.append(content_dict[key])
        assert(len(cluster) == len(indexes) + 1), "something is wrong"

    #model_selected = model.prediction_model(method="DecisionTree")
    model_selected = model.prediction_model(method="elasticnet",  max_iter=10000)
    mre = model.model_mre(model_selected, train_independent, train_dependent, test_independent, test_dependent)
    from numpy import median
    return round( median(mre)*100, 3), len(train_dependent)

def number_of_lines(filename):
    content = read_csv(filename)
    return len(content)

def run_experiment1():
    repeats = 20
    dir = "./Raw_Data/"
    filenames = sorted([f for f in listdir(dir) if isfile(join(dir, f)) and "ds101" in f])
    for filename in filenames:
        scores = []
        len_data = []
        for _ in list(range(repeats)):
            score, len = experiment(filename)
            scores.append(score)
            len_data.append(len)

        from numpy import median, percentile
        print(filename, median(scores), percentile(scores, 75) - percentile(scores, 25), median(len_data))


if __name__ == "__main__":
    # print "filename min max mean median Data(%)"
    run_experiment1()
