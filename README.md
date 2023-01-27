# Naive Bayes Classifier

**This program implements a Naive Bayes Classifier.**

## How It Works

A true Bayes Optimal Classifier works by using Bayes Theorem to calculate conditional probabilities, which could then be used to make the most probable prediction for any given test object. However, because the Bayes Classifier requires the calculation of the probability function P(X_1, ..., X_d | C_k), where all attributes X_i of object X are all dependent on each other, and C_k is any output class, it is often infeasible implement in practice.

However, by using a Naive Bayes Classifier (instead of a true Bayes Classifier), we avoid the problem of finding the complex, high-dimensional probability functions P(X_1, ..., X_d | C_k).

To do this, we assume that each attribute within an object X is independent of each other, allowing us to first model each attribute with a separate probability P(X_i | C_k), and then apply the product rule to find the joint probability:

P(X | C_k) = P(X_1, ..., X_d | C_k) = P(X_1 | C_k) * P(X_2 | C_k) * ... * P(X_d | C_k)

With P(X | C_k), we can infer upon any test object by applying Bayes Rule. This, as a reminder, is:

P(A|B) = P(B|A) * P(A) / P(B)

Given this, we can compute P(C_k | X) if we know P(X | C_k), P(X), and P(C_k). P(C_k) is a simple fraction of objects belonging to class C_k. P(X | C_k) can be computed using the product rule of independent attribute probability distributions, as shown earlier. P(X) can be computed using the sum rule over all classes C_k.

## Implementation

**Training:**\
For every attribute A, for every class C:
- Identify all training objects that belong to class C
- Compute the mean and standard deviation among all of the selected training objects. This defines the Gaussian for the density P(A|C) of attribute A, given class C.

**Test:**\
For every test object X = [X_1, ..., X_d]:\
&nbsp;&nbsp;&nbsp;&nbsp;For every class C_j in C_1, ..., C_k:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For every attribute value X_i in X_1, ..., X_d:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Compute p(X_i | C_j). This is a Gaussian density.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Using the Product Rule, compute P(X | C_j)\
&nbsp;&nbsp;&nbsp;&nbsp;Using the Sum Rule, compute P(X)\
&nbsp;&nbsp;&nbsp;&nbsp;For every class C_j in C_1, ..., C_k:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Compute P(C_j | X)\
&nbsp;&nbsp;&nbsp;&nbsp;Find the class Cj that maximizes P(Cj | X). This is the output of
the Naïve Bayes Classifier for test input X

## Installation

Clone the project repository onto your local machine via HTTPS or SSH. 

**HTTPS:**

```bash
git clone https://github.com/quynh16/Naive-Bayes.git
```

**SSH:**
```bash
git clone git@github.com:quynh16/Naive-Bayes.git
```

## Usage

Provide a path to a training and test file. 

The files must be formatted such that each object is on a single row, and each attribute of the object is a space-separated floating-point number in the range [0, 1]. The last column of every row is the object's label, which can be any string. Every object must have the same number of attributes as well as a class label, which should contain no spaces.

A set of sample training and test files is provided in the 'datasets' directory, all of which can also be used to train on and test the model.

To run the program from the command line:

```bash
python3 naive_bayes.py [training_file_path] [test_file_path]
```

## Sample Output

Sample output for pendigits_training.txt and pendigits_test.txt
![Sample output for pendigits_training.txt and pendigits_test.txt](https://github.com/quynh16/Naive-Bayes/blob/main/output.png?raw=true)

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
