'''
Quynh Nguyen
1001791420
'''

import sys
import numpy as np

'''Implementation of a Naive Bayes Classifier. This program takes in 2 command
line arguments, a training and test file, formatted as a matrix table where
each row is a test object, every column except the last one is a value of an
attribute of that object, and the last column is the class of that object.
Each attribute value is normalized to a value between 0 and 1. 

The training stage uses the training data to calculate the probabilities for
the model, and the inference stage uses those probabilities to predict which
class each test object belongs to.
'''

class DataReader():
    '''Class that helps parse data, which are read from the command line, into
     a training and test data set, where the data comes from a text file in 
    table format, with all columns except the last one being values/attributes 
    of an object, and the last column being the class of an object. Each row is 
    an object.
    '''

    def get_files_from_cmd(self):
        '''Parses command-line args and returns the training and test file 
        names.'''

        if len(sys.argv) <= 2:
            print("Error: No training and test file specified.")
            exit(1)
        elif len(sys.argv) == 3:
            return sys.argv[1], sys.argv[2]
        else:
            print("Error: Invalid arguments.")
            exit(1)

    def read_file(self, file):
        '''Reads in a string of the file name (file) and returns a list of rows,
        where each row is a list of strings.
        '''

        rows = []

        try:
            f = open(file, 'r')
        except IOError:
            print("Error: File does not exist.")
            exit(1)

        lines = f.readlines()
        for line in lines:
            rows.append(line.split())

        f.close()
        return rows

    def parse_data(self, data):
        '''Takes in a matrix and converts all values except the last column into
        a float. The last column represents the class label.
        '''

        parsed_data = []
        for object in data:
            obj = []

            # convert all columns except last one to float
            for col in range(len(object) - 1):
                obj.append(float(object[col]))

            # last column can stay as string
            obj.append(object[-1])
            parsed_data.append(obj)

        return parsed_data

    def get_class_list(self, objects):
        '''Takes a list of objects and returns a dictionary of {'Class C',
        [ClassStats]}, where [ClassStats] is a list of ClassStats objects.
        '''

        classes = {}
        num_objects = 0

        for object in objects:
            o_class = object[-1]

            if not o_class in classes:
                classes[o_class] = ClassStats(o_class)

            classes[o_class].add_object(object)
            num_objects += 1

        return classes, num_objects



class ClassStats:
    '''Class that stores information about all of the objects that belong to a
    certain class, including the class name, list of objects, statistics
    (parameters) used to create the model that classifies each object, into that
    specific class, etc.
    '''

    def __init__(self, name):
        self.name = name # name of the class
        self.num_objects = 0 # total num of objects in this class
        self.num_dims = 0 # num of dimensions in  object (not including class)
        self.objects = [] # list of objects (a list of floats and one string)
        self.stats = {} # dictionary of statistics for the class

    def add_object(self, object):
        '''Adds an object to the class's list of objects.
        Input: an object, i_object, represented as a list of values, where its
        last value is the object's class name.
        '''

        self.objects.append(object)
        self.num_objects += 1

    def set_stats(self, stat, value):
        self.stats[stat] = value

    def get_objects(self):
        return self.object_list

    def get_stats(self):
        return self.stats

    def get_stat(self, stat):
        return self.stats[stat]

    def get_num_objects(self):
        return self.num_objects

    def calculate_stats(self, stat_func):
        '''Calculates all of the required parameters to create a model that
        classifies objects into this class.
        Input: a helper class that does the calculations.
        '''

        self.num_dims = len(self.objects[0]) - 1
        self.stats.update(stat_func.calculate_stats(self.objects, 
                          self.num_objects, self.num_dims))

    def print_stats(self, stat_func):
        '''Prints the statistics required for a specific model.
        Input: a helper class that prints the statistics for a specific model
        '''

        stat_func.print_stats(self.name, self.stats, self.num_dims)



class GaussianStats():
    '''Helper class that creates a Gaussian distribution for each column
    (attribute/dimension) in the given list of objects (which all belong to a
    the same class) by calculating the mean and standard deviation (using the
    formula that divides by n - 1) for each column. Also helps with evaluating
    a Gaussian function given the mean, standard deviation, and a value x.
    '''

    def calc_means(self, objects, num_objects, num_dims):
        '''Calculates the means of all the columns in the list of objects'''

        num_dims = len(objects[0]) - 1
        mean_list = [0.0] * num_dims

        for object in objects:
            for dim in range(num_dims):
                mean_list[dim] += object[dim]

        for dim in range(num_dims):
            mean_list[dim] /= num_objects

        return mean_list

    def calc_stds(self, objects, num_objects, num_dims, means):
        '''Calculates the standard deviation of all the columns in the list of
        objects.'''

        std_list = [0.0] * num_dims

        for object in objects:
            for dim in range(num_dims):
                std_list[dim] += (object[dim] - means[dim]) ** 2

        for dim in range(num_dims):
            variance = std_list[dim] / (num_objects - 1)
            std_list[dim] = variance ** 0.5
            # std should never be < 0.01
            if std_list[dim] < 0.01:
                std_list[dim] = 0.01

        return std_list

    def calculate_stats(self, objects, num_objects, num_dims):
        '''Takes in a list of objects belonging to a certain class and 
        calculates the mean and standard deviation for each column in an object.
        '''

        mean_list = self.calc_means(objects, num_objects, num_dims)
        std_list = self.calc_stds(objects, num_objects, num_dims, mean_list)

        return {'mean': mean_list, 'std': std_list}

    def print_stats(self, class_name, stats, num_dims):
        '''Prints the mean and standard deviation for each column in the list
        of objects that belong to this specific class.'''

        for dim in range(num_dims):
            print('Class %s, attribute %d, mean = %.2f, std = %.2f' % 
            (class_name, dim + 1, stats['mean'][dim], stats['std'][dim]))

    def evaluate(self, i, x_i, stats):
        '''Evaluates a Gaussian at x_i with a given mean and standard deviation.
        '''

        mean = stats['mean'][i]
        std = stats['std'][i]

        expon = ((x_i - mean) ** 2) / (2 * (std ** 2))
        denom = std * ((2 * np.pi) ** 0.5)
        return ((np.e) ** (-expon)) / denom



class NaiveBayesClassifier():
    '''Classification model that predicts the class of a test object using
    probabilities calculated with Bayes Rule in the training stage. This model
    naively assumes that all dimensions of an object are independent of 
    eachother.
    '''

    def __init__(self, classes, total_objects, stat_func):
        self.classes = classes
        self.total_objects = total_objects
        self.stat_func = stat_func
        self.calc_p_C()

    def train(self):
        '''Trains the model by calculating the Gaussian of P(x|class) for each
        dimension for each class using the mean and std from the training data.
        '''

        for class_name in sorted(self.classes):
            # C = ClassStat object that contains a list of all objects in a
            #     certain class
            C = self.classes[class_name]
            C.calculate_stats(self.stat_func)
            C.print_stats(self.stat_func)

    def test(self, test_objects):
        '''Inferences on the test_objects (list of values, where the last value
        is the actual class), by predicting the class using Bayes Theorem. Also
        prints the results of the prediction and accuracy of the model.

        Input: a list of test objects
        '''

        id = 1 # object id
        total_accuracy = 0 # num of correct predictions
        total_objects = 0 # num of test objects

        for object in test_objects:
            accuracy = 0
            total_objects += 1
            actual_class = object[-1]

            # predict the class for the current object
            prob, pred_classes = self.classify(object)
            pred_class = pred_classes[0] # if tie, just select first class

            # if tie, accuracy = 1 / num of classes that tied for best
            if actual_class in pred_classes:
                accuracy = 1 / len(pred_classes)

            total_accuracy += accuracy

            self.print_results(id, pred_class, prob, actual_class, accuracy)
            id += 1

        # overall accuracy = average of all accuracies
        model_accuracy = total_accuracy / total_objects
        print('classification accuracy=%6.4f' % model_accuracy)

    def classify(self, X):
        '''Classifies an object X into a class C by selecting the class with the
        maximum probability using Baye's Rule to compute P(C_k | x) for every 
        possible output class C_k, while making the naive assumption that all 
        dimensions of X are independent of eachother.

        Input: a test object (list of values, where the last value is the class).
        '''

        max_prob = 0
        max_class = []

        all_p_X_given_C = self.calc_all_p_X_given_C(X)
        p_X = self.calc_p_X(X, all_p_X_given_C)

        for class_name in self.classes:
            C = self.classes[class_name]
            p_X_given_C = all_p_X_given_C[class_name]
            p_C = C.get_stat('p_of_C')

            # apply Bayes Rule
            # prob = P(C given X)
            prob = (p_X_given_C * p_C) / p_X

            # keep track of the max probability
            if prob > max_prob:
                max_prob = prob
                max_class = [class_name]
            elif prob == max_prob:
                max_class.append(class_name)

        return max_prob, max_class

    def print_results(self, id, pred_class, prob, actual_class, accuracy):
        '''Prints the result of one inference on a test object.'''

        print('ID=%5d, predicted=%3s, probability = %.4f, true=%3s, accuracy=%4.2f'
              % (id, pred_class, prob, actual_class, accuracy))

    def calc_p_C(self):
        '''Calculates the probability of each class in the training data by
        using the frequentist approach.
        --> p(C)
        '''

        for class_name in self.classes:
            C = self.classes[class_name]
            p_C = C.get_num_objects() / self.total_objects
            C.set_stats('p_of_C', p_C)

    def calc_p_Xi_given_C(self, C, i, X_i):
        '''Calculates the probability density of dimension i of X (X_i) given
        the current class C, using the respective statistics function.
        --> P(X_i | C)

        Input: a class (ClassStat C), a dimension (int i), and the value
        of that dimension in object X (float X_i)
        '''

        return self.stat_func.evaluate(i, X_i, C.get_stats())

    def calc_p_X_given_C(self, X, C):
        '''Calculates the probability density of object X given a class C using
        the Naive Bayes assumption that all dimensions are independent.
        --> P(X | C)

        Input: a test object (list X) and class (ClassStat C)
        '''

        p_X_given_C = 1

        # p(X|C) = basic product rule of all of X's dimensions
        for dim in range(len(X) - 1):
            p_X_given_C *= self.calc_p_Xi_given_C(C, dim, X[dim])

        return p_X_given_C

    def calc_all_p_X_given_C(self, X):
        '''Calculates the probability of X given C for all classes.
        --> P(X | C) for all C

        Input: a test object (list X)
        '''

        all_p_X_given_C = {}

        for class_name in self.classes:
            C = self.classes[class_name]
            p_X_given_C = self.calc_p_X_given_C(X, C)
            all_p_X_given_C[class_name] = p_X_given_C

        return all_p_X_given_C

    def calc_p_X(self, X, all_p_X_given_C):
        '''Calculates the probability of X using the sum rule across class C.
        --> P(X)

        Input: a test object (list X) and list of probabilities P(X | C) for all
        class C
        '''

        p_X = 0

        for class_name in self.classes:
            C = self.classes[class_name]
            p_X += all_p_X_given_C[class_name] * C.get_stat('p_of_C')

        return p_X



def main():
    dr = DataReader()
    training_file, test_file = dr.get_files_from_cmd()

    # parsing data from training text files
    training_data = dr.read_file(training_file)
    training_objects = dr.parse_data(training_data)

    # separating objects into lists for each class
    classes, total_training_objects = dr.get_class_list(training_objects)

    # create a Naive Bayes Classifier based on the training objects and stats
    # function used to calculate the probabilities for Bayes Rule

    # can modify the type of stat_func as long as it has a "calculate_stats()"
    # function that returns a float 
    stat_func = GaussianStats()
    nb = NaiveBayesClassifier(classes, total_training_objects, stat_func)

    # train the model by pre-calculating Gaussians for each dim & class in given
    # input training_file
    nb.train()

    # parsing data from testing text files
    test_data = dr.read_file(test_file)
    test_objects = dr.parse_data(test_data)

    # inferencing / testing on input test_file
    nb.test(test_objects)



if __name__ == '__main__':
    main()
