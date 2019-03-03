# Constructing a decision tree

I have implemented the code for constructing a decision tree, given a csv file of all the data points and their corresponding labels. The labels must be in the *last column*.

## Getting Started

Download inspect.py and decisionTree.py

### Prerequisites

The aim was to not use any fancy packages. So there are no prerequisites. This program only uses packages included in Python 2.7's standard library. All you need is a datasets on which you want to construst the tree.

### Running the program

inspect.py takes in 2 command line arguments. The first argument is the path to the file that we want to inspect. The second is the name of the file we want inspect to output to. For example:

```
$ python inspect.py training_data.csv inspected_data.txt
```

The point of inspect is to give the user the error rate without using a decision tree. The decisionTree.py's error rate should be better than the error output by inspect.py

decisionTree.py takes in 6 command-line arguments: <train input> <test input> <max depth> <train out> <test out> <metrics out>.

1. <train input>: path to the training input .csv file
2. <test input>: path to the test input .csv file
3. <max depth>: maximum depth to which the tree should be built
4. <train out>: path of output .labels file to which the predictions on the training data should be written
5. <test out>: path of output .labels file to which the predictions on the test data should be written
6. <metrics out>: path of the output .txt file to which metrics such as train and test error should be written 
  
For example:

```
 $ python decisionTree.py train.csv test.csv 2 train.labels test.labels metrics.txt
```

Additionally the tree is also printed to the std out, so that the user can visualize how it looks.
