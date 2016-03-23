from __future__ import division
from os import listdir
from os.path import isfile, join
from numpy import median


class data_item():
    def __init__(self, id, decisions, objective):
        self.id = id
        self.decisions = decisions
        self.objective = objective

    def __repr__(self):
        return str(self.id)+ "|" +",".join(map(str, self.decisions)) + "|" + str(self.objective)


def where_data_transformation(filename):
    from Utilities.WHERE.where import where
    # The Raw_Data has to be access using this attribute table._rows.cells
    import pandas as pd
    df = pd.read_csv(filename)
    headers = [h for h in df.columns if '$<' not in h]
    data = df[headers]
    clusters = where(data)
    return clusters

def random_clustering(filename):
    content = read_csv(filename)
    no_clusters = 4
    from random import random, shuffle
    clusters_size = [random() for _ in xrange(no_clusters)]
    clusters_size = [i/sum(clusters_size) for i in clusters_size]
    shuffle(content)
    clusters_size = [sum(clusters_size[:i]) for i in xrange(len(clusters_size))] +[1]
    clusters_size = [int(c * len(content)) for c in clusters_size]
    clusters = [[c.decisions for c in content[clusters_size[i]:clusters_size[i+1]]] for i in xrange(len(clusters_size) - 1)]
    return clusters


def transform(filename):
    return "./NormalizedData/" + filename


def read_csv(filename, header=False):
    def transform(filename):
        return "./NormalizedData/" + filename

    import csv
    data = []
    f = open(transform(filename), 'rb')
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



def localsearch(percentage, filename):

    content = read_csv(filename)
    content_dict = {}
    for c in content:
        key = ",".join(map(str, c.decisions))
        content_dict[key] = float(c.objective)

    mres = []
    clusters = random_clustering(filename)
    for cluster in clusters:
        length_of_data = cluster
        no_of_points = min(3, len(cluster))
        from random import sample
        train_idx = sample(xrange(len(cluster)), no_of_points)
        test_idx = [i for i in xrange(len(cluster)) if i not in train_idx]

        train_independent = [cluster[i] for i in train_idx]
        train_dependent = [content_dict[",".join(map(str, point))] for point in train_independent]

        test_independent = [cluster[i] for i in test_idx]
        test_dependent = [content_dict[",".join(map(str, point))] for point in test_independent]

        # print len(train_dependent), len(test_dependent)

        from sklearn import tree
        clf = tree.DecisionTreeRegressor()
        clf.fit(train_independent, train_dependent)
        predicted_dependent = clf.predict(test_independent)


        mres.append(median([abs(a-p)/(a+0.00001) for a, p in zip(predicted_dependent, test_dependent)])*100)

    return mres

if __name__ == "__main__":
    dir = "./NormalizedData/"
    percentage = 10
    filenames = [f for f in listdir(dir) if isfile(join(dir, f))]
    for filename in filenames:
        print filename,
        mres = localsearch(percentage, filename)
        print  median(mres)

