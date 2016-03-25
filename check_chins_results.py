from __future__ import division
from numpy import mean, median, percentile

class data_item():
    def __init__(self, id, decisions, objective):
        self.id = id
        self.decisions = decisions
        self.objective = objective

    def __repr__(self):
        return str(self.id)+ "|" +",".join(map(str, self.decisions)) + "|" + str(self.objective)

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


def test_type1(training_filename, testing_filename):
    mres = []
    for _ in xrange(10):
        training_content = read_csv(training_filename)
        train_independent = [t.decisions for t in training_content]
        train_dependent = [t.objective for t in training_content]

        all_testing_content = read_csv(testing_filename)
        from random import shuffle
        shuffle(all_testing_content)
        testing_content = all_testing_content[:int(len(all_testing_content) * 0.4)]
        test_independent = [t.decisions for t in testing_content]
        test_dependent = [t.objective for t in testing_content]

        from sklearn import tree
        clf = tree.DecisionTreeRegressor()
        # print len(train_dependent),
        clf.fit(train_independent, train_dependent)
        predicted_dependent = clf.predict(test_independent)
        mres.append(mean([abs(a-p)/(a+0.00001) for a, p in zip(predicted_dependent, test_dependent)]))
    return mres


def runner_test_type1():
    from os import listdir
    from os.path import isfile, join
    folder = "./Final/"
    filenames = [f for f in listdir(folder) if isfile(join(folder, f))]

    for filename in filenames:
        mres =  test_type1("./BestCase/" + filename, "./Final/"+filename)
        print median(mres), percentile(mres, 75) - percentile(mres, 25)


runner_test_type1()

