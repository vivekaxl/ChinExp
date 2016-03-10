from __future__ import division

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
            mre.append(abs(i - j) / float(i))
        else:
            if i==j: mre.append(0)

    return mre


def run_experiment1():
    filenames = ["1_tp_read.csv", "2_tp_write.csv", "3_tp_read.csv", "4_tp_write.csv"]
    for filename in filenames:
        # for easy look up
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

        mre = model_cart(train_independent, train_dependent, test_independent, test_dependent)
        data_percentage = round((len(train_dependent)/len_data) * 100, 3)
        from numpy import mean, median
        print filename, round(min(mre)*100, 3), round(max(mre)*100, 3), round(mean(mre)*100, 3),round( median(mre)*100, 3), data_percentage


if __name__ == "__main__":
    print "filename| min | max | mean | median | Data(%)"
    run_experiment1()