import numpy as np
import pickle as pk


class DecisionNode():
    def __init__(self):
        self.X, self.Y = None, None
        self.children = dict()
        self.cond = None


    def __entropy(self, arr):
        probs = [np.sum(arr==x) / arr.shape[0] for x in set(arr)]
        return -sum([p*np.log2(p) for p in probs])


    def __min_entropy_split(self):
        entropies = []
        for i in range(self.X.shape[1]):
            for v in set(self.X[:, i]):
                labels1 = self.Y[self.X[:, i] == v]
                labels2 = self.Y[self.X[:, i] != v]
                p = np.sum(self.X[:, i]==v) / self.X.shape[0]
                
                entropy = p * self.__entropy(labels1)
                entropy += (1 - p) * self.__entropy(labels2)
                entropies.append((entropy, i, v))
        i, v = self.cond = min(entropies)[1:]
        
        cond1, cond2 = (self.X[:, i] == v), (self.X[:, i] != v)
        return self.X[cond1], self.X[cond2], self.Y[cond1], self.Y[cond2]
 

    def expand(self, X, Y):
        self.X, self.Y = X, Y
        if len(set(Y)) == 1:
            return
        
        X1, X2, Y1, Y2 = self.__min_entropy_split()
        if X1.shape[0] == 0 or X2.shape[0] == 0:
            self.cond = None
            return

        node1, node2 = DecisionNode(), DecisionNode()
        node1.expand(X1, Y1)
        node2.expand(X2, Y2)

        self.children[self.cond] = node1
        self.children['<OTHER>'] = node2


    def __str__(self):
        string = '-' * 32 + '\n'
        if not len(self.children):
            string = string + '\n**LEAF NODE**'
        return string + '\n{}, {}\n'.format(self.X, self.Y)
        


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
                node = node.children[node.cond]
            else:
                node = node.children['<OTHER>']
        return max([(np.sum(node.Y==x), x) for x in set(node.Y)])[1]


    def pickle(self, fname):
        objects = [self, self.root]
        queue = [self.root]
        while len(queue):
            node = queue.pop(0)
            objects.append(node)
            for child in node.children.values():
                queue.append(child)
        with open(fname, 'wb') as f:
            pk.dump(objects, f)
            







###################################################
# TEST
###################################################

X = ['AA +', 'AA -', 'AA +', 'AA +',
     'BB +', 'BB -', 'CC +', 'CC -']
X = np.array([x.split(' ') for x in X])
Y = np.array([1, 0, 1, 0, 0, 0, 0, 0])


#tree = BinaryDecisionTree()
#tree.fit(X, Y)

with open('tree.pk', 'rb') as f:
   objects = pk.load(f)
   tree = objects[0]

x = np.array(['AA', '+'], dtype=X.dtype)
print(tree.predict(x))

tree.pickle('tree.pk')
