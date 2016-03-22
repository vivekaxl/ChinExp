import numpy as np
import matplotlib.pyplot as plt


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


def feature_important(filename):
    from sklearn.datasets import make_classification
    from sklearn.ensemble import ExtraTreesClassifier

    content = read_csv(filename)
    X = [c.decisions for c in content]
    y = [c.objective for c in content]

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    #
    for f in range(len(X[0])):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(X[0])), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(X[0])), indices)
    plt.xlim([-1, len(X[0])])
    plt.show()


# feature_important("1_tp_read.csv")
# feature_important("ds101_rt_read.csv")
feature_important("ds101_rt_write.csv")