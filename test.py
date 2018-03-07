from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import loader
import pylab
import grapher
import time

def main():

    data_load = loader.MNIST(return_type='numpy')
    testing_im, testing_lb = data_load.load_testing()
    print testing_lb

main()