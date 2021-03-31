from sklearn import datasets
import sys
import pca
import kpca
import lda


# Press the green button in the gutter to run the script.


def main():
    if len(sys.argv) != 3:
        print("Wrong number of arguments")
        sys.exit(0)
    elif sys.argv[2] == 'digits':
        dataset = datasets.load_digits(as_frame=True, return_X_y=True)
    elif sys.argv[2] == ''



if __name__ == '__main__':
    main()
