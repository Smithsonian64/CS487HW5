from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class lda:
    def __init__(self, dataset):
        self.dataset = dataset

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset[0], self.dataset[1],
                                                                                test_size=0.3, random_state=109)

        self.classifier = DecisionTreeClassifier()
        self.classifier.fit(self.X_train, self.y_train)

        start_time = time.time()
        self.y_pred = self.classifier.predict(self.X_test)

        print("running time for dt: %s seconds " % (time.time() - start_time))
        print("Accuracy for dt:", accuracy_score(self.y_test, self.y_pred))

        sc = StandardScaler()
        self.X_train_std = sc.fit_transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)
        self.cov_mat = np.cov(self.X_train_std.T)
        self.eigen_vals, self.eigen_vecs = np.linalg.eig(self.cov_mat)
        self.eigen_pairs = [(np.abs(self.eigen_vals[i]), self.eigen_vecs[:, i]) for i in range(len(self.eigen_vals))]
        self.eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        w = np.hstack((self.eigen_pairs[0][1][:, np.newaxis], self.eigen_pairs[1][1][:, np.newaxis]))
        self.x_train_lda = self.X_train_std.dot(w)

        l = LDA()
        self.X_train_lda = l.fit_transform(self.X_train_std, self.y_train)

        tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
        tree_model.fit(self.X_train_lda, self.y_train)

        self.X_test_lda = l.transform(self.X_test_std)
        self.y_pred_lda = tree_model.predict(self.X_test_lda)


        print("running time for dt+lda: %s seconds " % (time.time() - start_time))
        print("Accuracy for dt+lda:", accuracy_score(self.y_pred_lda, self.y_test))

        print()