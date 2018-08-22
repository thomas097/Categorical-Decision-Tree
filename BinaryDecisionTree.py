import numpy as np
import pickle as pk


class DecisionNode():
    def __init__(self):
        self.children = None
        self.cond = None
        self.label = None


    def __entropy(self, arr):
        probs = [np.sum(arr==x) / arr.shape[0] for x in set(arr)]
        return -sum([p*np.log2(p) for p in probs])


    def __min_entropy_split(self, X, Y):
        entropies = []
        for i in range(X.shape[1]):
            for v in set(X[:, i]):
                labels1, labels2 = Y[X[:, i] == v], Y[X[:, i] != v]
                p = np.sum(X[:, i]==v) / X.shape[0]

                # calculate entropy given feature and value
                entropy = p * self.__entropy(labels1)
                entropy += (1 - p) * self.__entropy(labels2)
                entropies.append((entropy, i, v))
        i, v = self.cond = min(entropies)[1:]

        # perform data split and return the split set
        cond1, cond2 = (X[:, i] == v), (X[:, i] != v)
        return X[cond1], X[cond2], Y[cond1], Y[cond2]
 

    def expand(self, X, Y):
        self.label = max([(np.sum(Y==y), y) for y in set(Y)])[1]

        # do not split a singleton set
        if len(set(Y)) == 1:
            return

        # stop when one of the sets is empty
        X1, X2, Y1, Y2 = self.__min_entropy_split(X, Y)
        if X1.shape[0] == 0 or X2.shape[0] == 0:
            self.cond = None
            return

        # create branches recursively
        node1, node2 = DecisionNode(), DecisionNode()
        node1.expand(X1, Y1)
        node2.expand(X2, Y2)
        self.children = (node1, node2)
        


class BinaryDecisionTree():
    def __init__(self):
        self.root = DecisionNode()


    def fit(self, X, Y):
        self.root.expand(X, Y)


    def predict(self, x):
        node = self.root
        while node.cond:
            i, v = node.cond
            if x[i] == v:
                print(node.cond, 'yes')
                node = node.children[0]
            else:
                print(node.cond, 'no')
                node = node.children[1]
        return node.label


    def pickle(self, fname):
        objects = [self, self.root]
        queue = [self.root]
        while len(queue):
            node = queue.pop(0)
            objects.append(node)
            if node.children:
                for child in node.children:
                    queue.append(child)
        with open(fname, 'wb') as f:
            pk.dump(objects, f)
            







###################################################
# TEST
###################################################

X = ['AA +', 'AA -', 'AA ?', 'AA +',
     'BB +', 'BB -', 'CC +', 'CC -', 'AA 9']
X = np.array([x.split(' ') for x in X])
Y = np.array([1, 0, 1, 0, 0, 0, 1, 0, 2])


tree = BinaryDecisionTree()
tree.fit(X, Y)

#with open('tree.pk', 'rb') as f:
#   objects = pk.load(f)
#   tree = objects[0]

x = np.array(['AA', '-'], dtype=X.dtype)
print(tree.predict(x))

tree.pickle('tree.pk')
