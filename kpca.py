from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class kpca:
    def __init__(self, dataset):
        self.dataset = dataset

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset[0], self.dataset[1],
                                                                                test_size=0.3, random_state=109)

        self.classifier = DecisionTreeClassifier()
        self.classifier.fit(self.X_train, self.y_train)

        start_time = time.time()
        self.y_pred = self.classifier.predict(self.X_test)

        print("running time for kpca: %s seconds " % (time.time() - start_time))
        print("Accuracy for dt:", accuracy_score(self.y_test, self.y_pred))

        sc = StandardScaler()
        self.X_train_std = sc.fit_transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)
        self.cov_mat = np.cov(self.X_train_std.T)
        self.eigen_vals, self.eigen_vecs = np.linalg.eig(self.cov_mat)
        self.eigen_pairs = [(np.abs(self.eigen_vals[i]), self.eigen_vecs[:, i]) for i in range(len(self.eigen_vals))]
        self.eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        w = np.hstack((self.eigen_pairs[0][1][:, np.newaxis], self.eigen_pairs[1][1][:, np.newaxis]))
        self.x_train_kpca = self.X_train_std.dot(w)


        start_time = time.time()
        k = KernelPCA(kernel='rbf', gamma=15)
        self.X_kpca = k.fit_transform(self.X_train)
        self.X_train_kpca = k.fit_transform(self.X_train)
        self.X_test_kpca = k.transform(self.X_test)

        tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
        tree_model.fit(self.X_train_kpca, self.y_train)


        self.y_pred_kpca = tree_model.predict(self.X_test_kpca)


        print("running time for dt+kpca: %s seconds " % (time.time() - start_time))
        print("Accuracy for dt+kpca:", accuracy_score(self.y_pred_kpca, self.y_test))

        print()