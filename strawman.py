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


def number_of_lines(filename):
    content = read_csv(filename)
    return len(content)


def strawman(split_point, filename):
    content = read_csv(filename)
    indexes = range(len(content))
    from random import shuffle
    shuffle(indexes)

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


def runner(filename, percentages):
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
        counts.append(percentage)

    for i in xrange(len(percentages)):
        print filename, percentages[i], median_result[i], iqr_results[i], counts[i]


if __name__ == "__main__":
    from random import seed
    seed(1023)
    dir = "./Raw_Data/"
    filenames = [f for f in listdir(dir) if isfile(join(dir, f))]
    samples = {}
    samples["1_tp_read.csv"] = [32*i for i in xrange(1, 7)]
    samples["2_tp_write.csv"] = [32*i for i in xrange(1, 7)]
    samples["3_tp_read.csv"] = [64*i for i in xrange(1, 7)]
    samples["4_tp_write.csv"] = [64*i for i in xrange(1, 7)]
    samples["ds101_ops_read.csv"] = [56*i for i in xrange(1, 7)]
    samples["ds101_ops_write.csv"] = [56*i for i in xrange(1, 7)]
    samples["ds101_rt_read.csv"] = [56*i for i in xrange(1, 7)]
    samples["ds101_rt_write.csv"] = [56*i for i in xrange(1, 7)]
    samples["ds101_tp_read.csv"] = [56*i for i in xrange(1, 7)]
    samples["ds101_tp_write.csv"] = [56*i for i in xrange(1, 7)]
    for filename in filenames:
        runner(filename, samples[filename])
