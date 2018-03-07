from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import loader
import pylab
import grapher
import time
import numpy as np

def main():
    average_error = [.118402352118,.106677842987,0.113631954756,0.124001470498,0.107012356003,0.14040359204,0.101725128759]
    axis = [0, 10, 0, 1]
    # pylab.title('Exact')
    # pylab.plot(zeros, 'bx', ones, 'gx', twos, 'rx',threes, 'cx',fours, 'mx',fives, 'yx',six, 'b+',seven, 'g+',eight, 'r+',nine, 'c+')
    pylab.plot(average_error, 'b')
    pylab.axis(axis)
    pylab.title("Error measure ")
    pylab.xlabel("average error: number of points 5000 + i * 750")
    pylab.ylabel("error measure")
    pylab.show()
    print("Retriving data set")
    data_load = loader.MNIST(return_type='numpy')
    training_im , training_lb = data_load.load_training()
    testing_im, testing_lb = data_load.load_testing()
    #out = None
    #zeros = []
    #ones  = []
    #twos = []
    #threes = []
    #fours = []
    #fives = []
    #six = []
    #seven = []
    #eight = []
    #nine = []
    average_error = []

    for i in range(10):
        final_error = []
        error = []
        num_points = 5000 + i * 750
        error_rate = 0.0

        print "starting TSNE n_components run"
        #this loop is to get the program to run on my computer.
        #It could be removed if there is a computer with enough RAM

        for t in range(10):
            training = []
            label = []
            testing = []
            test_lb = []

            idx = np.random.randint(len(training_im), size=len(training_im))
            idxt = np.random.randint(len(testing_im), size=len(testing_im))
            for i in range(num_points):
                training.append(training_im[idx[i]])
                label.append(training_lb[idx[i]])
                if num_points < len(testing_im) :
                    testing.append(testing_im[idxt[i]])
                    test_lb.append(testing_lb[idxt[i]])
            #if(i == 0) :
             #   print("will not go less then 2")
            #else :
            pca = PCA(n_components=50)
            out_pca = pca.fit_transform(training,label)
            print("Start TSNE")
            model = TSNE(n_components = 2)
            #start = time.time()
            out_train = model.fit_transform(out_pca,label)
            print("Model fit")
            out_test = model.fit_transform(testing)
            print("Model fit_transform")
            #end = time.time()
            #print("Run time of Barnes-Hut at")
            #print end - start
            neigh = KNeighborsClassifier(n_neighbors=5)
            neigh.fit(out_train,label)
            error = neigh.predict(out_test)
            for k in range(len(error)):
                if error[k] == test_lb[k] :
                    error_rate = error_rate + 1
            error_rate =  error_rate / num_points
            print error_rate
            final_error.append(error_rate)
        average_final = sum(final_error) / 10.0
        print average_final
        average_error.append(average_final)

            #times.append(end-start)
            #for point in range(len(out_t)) :
             #   if testing_lb[point] == 0 :
              #      zeros.append(out_t[point])
               # elif testing_lb[point] == 1:
                #    ones.append(out_t[point])
                #elif testing_lb[point] == 2:
                #    twos.append(out_t[point])
                #elif testing_lb[point] == 3:
                #    threes.append(out_t[point])
                #elif testing_lb[point] == 4:
                #    fours.append(out_t[point])
                #elif testing_lb[point] == 5:
                #    fives.append(out_t[point])
                #elif testing_lb[point] == 6:
                #    six.append(out_t[point])
                #elif testing_lb[point] == 7:
                #    seven.append(out_t[point])
                #elif testing_lb[point] == 8:
                #    eight.append(out_t[point])
                #else :
                #    nine.append(out_t[point])



main()