from __future__ import division
from os import listdir
from os.path import isfile, join
from numpy import median, mean


class data_item():
    def __init__(self, id, decisions, objective):
        self.id = id
        self.decisions = decisions
        self.objective = objective

    def __repr__(self):
        return str(self.id)+ "|" +",".join(map(str, self.decisions)) + "|" + str(self.objective)


def where_data_transformation(filename):
    from Utilities.WHERE.where import where
    # The Data has to be access using this attribute table._rows.cells
    import pandas as pd
    df = pd.read_csv(filename)
    headers = [h for h in df.columns if '$<' not in h]
    data = df[headers]
    clusters = where(data)
    return clusters


def read_csv(filename, header=False):

    import csv
    data = []
    f = open((filename), 'rb')
    reader = csv.reader(f)
    for i,row in enumerate(reader):
        if i == 0 and header is False: continue  # Header
        elif i ==0 and header is True:
            H = row
            continue
        data.append(data_item(i, map(float, row[:-1]), float(row[-1])))
    f.close()
    if header is True: return H, data
    return data

def euclidean_distance(list1, list2):
    assert(len(list1) == len(list2)), "The points don't have the same dimension"
    distance = sum([(i - j) ** 2 for i, j in zip(list1, list2)]) ** 0.5
    assert(distance >= 0), "Distance can't be less than 0"
    return distance



def localsearch(percentage, filename):
    content = read_csv(filename)
    content_dict = {}
    for c in content:
        key = ",".join(map(str, c.decisions))
        content_dict[key] = float(c.objective)

    mres = []
    clusters = where_data_transformation(transform(filename))

    for cluster in clusters:
        no_of_points = min(percentage, len(cluster))
        from random import sample
        train_idx = sample(xrange(len(cluster)), no_of_points)
        test_idx = [i for i in xrange(len(cluster)) if i not in train_idx]
        assert(len(train_idx) + len(test_idx) == len(cluster)), "Something is wrong"

        train_independent = [cluster[i] for i in train_idx]
        train_dependent = [content_dict[",".join(map(str, map(float, point)))] for point in train_independent]

        test_independent = [cluster[i] for i in test_idx]
        test_dependent = [content_dict[",".join(map(str, map(float, point)))] for point in test_independent]

        from sklearn import tree
        clf = tree.DecisionTreeRegressor()
        clf.fit(train_independent, train_dependent)
        predicted_dependent = clf.predict(test_independent)
        mres.append(median([abs(a-p)/(a+0.00001) for a, p in zip(predicted_dependent, test_dependent)])*100)
        # print len(cluster), len(train_dependent), len(test_dependent), round(mres[-1], 3)

    return mres


def localsearch2(filename, test_independent, test_dependent):
    import numpy as np
    """trying to find points to sample"""
    content = read_csv(filename)
    content_dict = {}
    for c in content:
        key = ",".join(map(str, c.decisions))
        content_dict[key] = float(c.objective)

    mres = []
    clusters = where_data_transformation( filename)

    train_independent = []
    train_dependent = []

    for cluster in clusters:
        distance_matrix = [[-1 for _ in xrange(len(cluster))] for _ in xrange(len(cluster))]
        for i in xrange(len(cluster)):
            for j in xrange(len(cluster)):
                if distance_matrix[i][j] == -1:
                    temp_dist = euclidean_distance(cluster[i], cluster[j])
                    distance_matrix[i][j] = temp_dist
                    distance_matrix[j][i] = temp_dist
        scores = [1/sum(distance_matrix[i])**2 for i in xrange(len(cluster))]
        mid_value = sorted(scores)[int(len(scores)/2)]
        train_index = [i for i,val in enumerate(scores) if val==mid_value][-1]
        train_independent.append(cluster[train_index])
        train_dependent.append(content_dict[",".join(map(str, map(float,cluster[train_index])))])
    from sklearn import tree
    clf = tree.DecisionTreeRegressor()
    print len(train_dependent),
    clf.fit(train_independent, train_dependent)
    predicted_dependent = clf.predict(test_independent)
    return mean([abs(a-p)/(a+0.00001) for a, p in zip(predicted_dependent, test_dependent)])


def random_sampling(filename, test_independent, test_dependent):
    content = read_csv(filename)
    sampling_percent = 20

    from random import shuffle
    shuffle(content)
    training = content[:int(len(content) * sampling_percent/100)]
    train_independent = [t.decisions for t in training]
    train_dependent = [t.objective for t in training]

    from sklearn import tree
    clf = tree.DecisionTreeRegressor()
    print len(train_dependent),
    clf.fit(train_independent, train_dependent)
    predicted_dependent = clf.predict(test_independent)
    return mean([abs(a-p)/(a+0.00001) for a, p in zip(predicted_dependent, test_dependent)])


def localsearch3(percentage, filename):
    content = read_csv(filename)
    content_dict = {}
    for c in content:
        key = ",".join(map(str, c.decisions))
        content_dict[key] = float(c.objective)

    mres = []
    clusters = where_data_transformation(transform(filename))

    for cluster in clusters:
        scores = [content_dict[",".join(map(str, map(float,c)))] for c in cluster]
        print min(scores)
        print max(scores)
        from numpy import mean, median
        print mean(scores)
        print median(scores)
        print sorted([round(s, 3) for s in scores])

    raw_input()
    return mres


if __name__ == "__main__":
    dir = "./Final/"
    percentages = [0.6]
    filenames = [f for f in listdir(dir) if isfile(join(dir, f))]
    methods = [localsearch2] #[localsearch2, random_sampling]
    for percentage in percentages:

        for filename in filenames:
            # print "---"*10
            # print filename
            for method in methods:
                # print method.__name__
                mres = []
                for _ in xrange(10):
                    # split into train and test
                    h, content = read_csv(dir+filename, header=True)
                    from random import shuffle
                    shuffle(content)
                    split_point = int(len(content) * percentage)

                    training = content[:split_point]
                    testing = content[split_point:]

                    # generate a temp file
                    temp_content = ""
                    temp_content += ",".join(h) + "\n"
                    for train in training:
                        temp_content += ",".join(map(str, train.decisions)) + "," + str(train.objective) + "\n"

                    f = open("/tmp/temp.csv", "w")
                    f.write(temp_content)
                    f.close()

                    mres.append(method("/tmp/temp.csv", [t.decisions for t in training], [t.objective for t in training]))
                    # print [round(m, 3) for m in mres],
                from numpy import percentile
                print round(median(mres), 3), round(percentile(mres, 75) - percentile(mres, 25), 3)


