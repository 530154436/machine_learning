# /usr/bin/env python3
# -*- coding:utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
def test():
    x_train = [[5,4],[9,6],[4,7],[2,3],[8,1],[7,2]]
    y_train = [1,1,1,0,0,0]

    x_test = [[5,3]]

    print('sklearn.neighbors.KNeighborsClassifier')
    for n_neighbors in range(1,6):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(x_train, y_train)
        y_pre = knn.predict(x_test)
        print("k = {}, y_pre = {}".format(n_neighbors, y_pre))

if __name__ == '__main__':
    test()
