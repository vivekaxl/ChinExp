from __future__ import division
from os import listdir
from os.path import isfile, join

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


def euclidean_distance(list1, list2):
    assert(len(list1) == len(list2)), "The points don't have the same dimension"
    distance = sum([(i - j) ** 2 for i, j in zip(list1, list2)]) ** 0.5
    assert(distance >= 0), "Distance can't be less than 0"
    return distance


def hotspot(points):
    distance_matrix = [[-1 for _ in xrange(len(points))] for _ in xrange(len(points))]
    for i,_ in enumerate(points):
        for j,_ in enumerate(points):
            if distance_matrix[i][j] == -1 and i!=j:
                temp = euclidean_distance(points[i], points[j])
                distance_matrix[i][j] = temp
                distance_matrix[j][i] = temp
            elif i == j: distance_matrix[i][j] = 0

    scores = [0 for _ in xrange(len(points))]
    for i,point in enumerate(points):
        scores[i] = sum([1/d**2 for j,d in enumerate(distance_matrix[i]) if i!=j] )

    return scores.index(max(scores))


from random import choice


def experiment(filename, sampling=None):
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
        indexes = range(len(cluster))
        random_point_index = sampling(cluster)
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

    mre = model_cart(train_independent, train_dependent, test_independent, test_dependent)
    from numpy import median
    return round( median(mre)*100, 3), len(train_dependent)

def number_of_lines(filename):
    content = read_csv(filename)
    return len(content)

def run_experiment1():
    repeats = 20
    dir = "./Raw_Data/"
    filenames = [f for f in listdir(dir) if isfile(join(dir, f))]
    for filename in filenames:
        scores = []
        len_data = []
        for _ in xrange(repeats):
            score, len = experiment(filename, hotspot)
            scores.append(score)
            len_data.append(len)

        from numpy import median, percentile
        print filename, median(scores), percentile(scores, 75) - percentile(scores, 25), median(len_data)



def strawman(percentage, filename):
    content = read_csv(filename)
    indexes = range(len(content))
    from random import shuffle
    shuffle(indexes)
    split_point = int(len(indexes) * percentage/100)

    train_indexes = indexes[:split_point]
    # test_indexes = indexes[:split_point]
    test_indexes = indexes[split_point:]
    # print "Trainset: ", len(train_indexes), " Testset: ", len(test_indexes), percentage, split_point

    train_set = [content[i] for i in train_indexes]
    test_set = [content[i] for i in test_indexes]

    train_independent = [t.decisions for t in train_set]
    train_dependent = [t.objective for t in train_set]

    test_independent = [t.decisions for t in test_set]
    test_dependent = [t.objective for t in test_set]

    mre = model_cart(train_independent, train_dependent, test_independent, test_dependent)

    from numpy import median
    return round(median(mre)*100, 3)


def runner(filename):
    # percentages =[10*i for i in xrange(1,9)]
    percentages =[ 10, 20, 30]
    repeat = 20
    results = []
    number_of_entries = number_of_lines(filename)
    median_result = []
    iqr_results = []
    counts = []
    from numpy import median, percentile
    for percentage in percentages:
        for _ in xrange(repeat):
            results.append(strawman(percentage, filename))
        median_result.append(median(results))
        iqr_results.append(percentile(results, 75) - percentile(results, 25))
        counts.append(percentage * number_of_entries/100)

    print filename
    for i in xrange(len(percentages)):
        print percentages[i], median_result[i], iqr_results[i], counts[i]


if __name__ == "__main__":
    # print "filename min max mean median Data(%)"
    run_experiment1()