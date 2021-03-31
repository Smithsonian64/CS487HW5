from sklearn import datasets
import sys
import pca
import kpca
import lda
import os, ssl


# Press the green button in the gutter to run the script.


def main():
    if len(sys.argv) != 3:
        print("Wrong number of arguments")
        sys.exit(0)
    elif sys.argv[2] == 'iris':
        dataset = datasets.load_iris(as_frame=True, return_X_y=True)
        dataset
    elif sys.argv[2] == 'mnist':
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
                getattr(ssl, '_create_unverified_context', None)):
            ssl._create_default_https_context = ssl._create_unverified_context

        dataset = datasets.fetch_openml('mnist_784', as_frame=True, return_X_y=True)

    if sys.argv[1] == 'pca':
        p = pca.pca(dataset)

    elif sys.argv[1] == 'lda':
        l = lda.lda(dataset)

    elif sys.argv[1] == 'kpca':
        k = kpca.kpca(dataset)



if __name__ == '__main__':
    main()
