# from collections import defaultdict
import argparse
import csv
import math
import collections

def parse_file():
    parser = argparse.ArgumentParser(description='Inspect a file')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    return args


def convert_csv_to_dict(file_name):
    # dictionary = defaultdict(list)
    dictionary = dict()
    label_name = None
    with open(file_name, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # row.iteritems() -> [(k,v), (k,v), ...]
            for key, value in row.iteritems():
                # this if block is not needed when using defaultdict
                if key not in dictionary:
                    dictionary[key] = []
                dictionary[key].append(value)
                if not label_name:
                    label_name = csv_reader.fieldnames[-1]
    list_of_label_features = dictionary[label_name]
    set_of_label_features = list(set(list_of_label_features))
    feature1_label = set_of_label_features[0]
    feature2_label = set_of_label_features[1]
    count_feature1 = list_of_label_features.count(feature1_label)
    count_feature2 = list_of_label_features.count(feature2_label)
    #print ("[%d %s/ %d %s]" % (count_feature1, feature1_label, count_feature2, feature2_label))
    return dictionary, label_name


# def convert_csv_to_dict(args):
#     # d = defaultdict(list)
#     d = {}
#     with open(args.input, mode='r') as csv_file:
#         csv_reader = csv.reader(csv_file)
#         headers = csv_reader.next()
#         label_name = headers[-1]
#         d[label_name] = []
#         for i,v in enumerate(headers):
#             d[v] = []
#         for row in csv_reader:
#             for i,v in enumerate(row):
#                 d[headers[i]] += [v]
#     return d, label_name


def entropy_of_label(dictionary, label_name):
    """
    :param dictionary: dictionary from csv
    :param label_name: last attribue. eg party
    :return:
    """
    list_of_label_features = dictionary[label_name]
    total_number_of_features = len(list_of_label_features)
    count_of_label_features = dict()#collections.defaultdict(int)

    for label_feature in list_of_label_features:
        if label_feature not in count_of_label_features:
            count_of_label_features[label_feature]= 0
        count_of_label_features[label_feature] += 1.0

    entropy = 0
    for _, label_feature_count in count_of_label_features.items():
        entropy += log_base2(label_feature_count, total_number_of_features)
    return entropy


def error_of_label(dictionary, label_name):
    label_feature_counter = collections.defaultdict(int)
    list_of_label_features = dictionary[label_name]
    for label_feature in list_of_label_features:
        label_feature_counter[label_feature] += 1
    majority_vote = max(label_feature_counter, key=label_feature_counter.get)
    error = 1 - (label_feature_counter[majority_vote]/float(len(list_of_label_features)))
    #print label_feature_counter[majority_vote]
    #print len(list_of_label_features)
    return error

def log_base2(numerator, denominator):
    """

    :param numerator:
    :param denominator:
    :return:
    """
    if numerator == 0 or denominator == 0:
        return 0
    x = float(numerator) / denominator
    y = math.log(x, 2)
    return -x * y

def write_file(args, entropy, error):
    with open(args.output, mode='w') as out_file:
        out_file.write("entropy: %.12f\nerror: %.12f" % (entropy, error))


def main():
    args = parse_file()
    dictionary, label_name = convert_csv_to_dict(args.input)
    entropy = entropy_of_label(dictionary, label_name)
    error = error_of_label(dictionary, label_name)
    write_file(args, entropy, error)

if __name__=='__main__':
    main()
