import math
import csv
import argparse
import sys
import collections


class Node():
    def __init__(self):
        self.name = ''
        self.count_dict = {}
        self.children = {}


class Decision():
    def __init__(self, name=''):
        self.name = name


def parse_command_line():
    parser = argparse.ArgumentParser(description='Read the command line')
    parser.add_argument('train_input')
    parser.add_argument('test_input')
    parser.add_argument('max_depth', type=int)
    parser.add_argument('train_output')
    parser.add_argument('test_output')
    parser.add_argument('metrics_output')
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


def split_children(n, mutual_information, current_depth, max_depth, dictionary, best_attribute, label_name,
                   attribute_feature):
    mod_dict = dictionary_modifier(dictionary, best_attribute, attribute_feature)
    if mutual_information > 0 and current_depth < max_depth and len(mod_dict) > 1:
        n.children[attribute_feature] = construct_decision_tree(mod_dict, max_depth, current_depth + 1, label_name)
    else:  # if current_depth >= max_depth, if mutual info <= 0:
        decision = max(n.count_dict[attribute_feature]['label_counts'], key=n.count_dict[attribute_feature]['label_counts'].get)
        n.children[attribute_feature] = Decision(name=decision)


def construct_decision_tree(dictionary, max_depth, current_depth, label_name):
    """
    :param dictionary:
    :param max_depth:
    :param current_depth:
    :param label_name:
    :return:
    """

    # if max_depth is 0
    if max_depth == 0:
        return majority_vote_classifier(dictionary, label_name)

    # there is at least 1 attribute in dict
    if len(dictionary) > 1:
        n = Node()
        # bar = "|"
        mutual_info_dict = mutual_info_dict_builder(dictionary, label_name)
        best_attribute = best_attribute_for_splitting(mutual_info_dict)
        mutual_information = mutual_info_dict[best_attribute]
        count_dict = get_count_dict_from_dictionary(dictionary, best_attribute, label_name)
        n.name = best_attribute
        n.count_dict = count_dict

        for attribute_feature in n.count_dict.keys():
            split_children(n, mutual_information, current_depth, max_depth, dictionary, best_attribute, label_name,
                           attribute_feature)
        return n


def majority_vote_classifier(dictionary, label_name):
    """ Returns majority decision based on label.

    :param dictionary: the dictionary representing data
    :param label_name: column name for label
    :param feature1_label: one of the features of label e.g. democrat
    :param feature2_label: the other feature of label e.g. republican
    :return: Decision object
    """
    label_feature_counter = collections.defaultdict(int)
    list_of_label_features = dictionary[label_name]
    for label_feature in list_of_label_features:
        label_feature_counter[label_feature] += 1
    majority_vote = max(label_feature_counter, key=label_feature_counter.get)
    return Decision(name=majority_vote)


def best_attribute_for_splitting(mutual_info_dict):
    """Calculate mutual info of all the attributes in given dictionary and pick attribute with max mutual info"""
    name = max(mutual_info_dict, key=mutual_info_dict.get)
    return name


def mutual_info_dict_builder(dictionary, label_name):
    """Calculate mutual info of all the attributes in the dictionary and store it in a temporary dict
    """
    mutual_info_dict = {}
    for attribute in dictionary.keys():
        # print attribute
        if not label_name == attribute:
            # print attribute
            mutual_info = calculate_mutual_info(dictionary, label_name, attribute)
            mutual_info_dict[attribute] = mutual_info
    return mutual_info_dict


def calculate_mutual_info(dictionary, label_name, attribute_name):
    """Calculate mutual info of one attribute in the dictionary"""
    conditional_entropy = calculate_conditional_entropy(dictionary, attribute_name, label_name)
    mutual_info = entropy_of_label(dictionary, label_name) - conditional_entropy
    return mutual_info


def entropy_of_label(dictionary, label_name):
    """
    :param dictionary: dictionary from csv
    :param label_name: last attribue. eg party
    :return:
    """
    list_of_label_features = dictionary[label_name]
    total_number_of_features = len(list_of_label_features)
    count_of_label_features = collections.defaultdict(int)

    for label_feature in list_of_label_features:
        count_of_label_features[label_feature] += 1

    entropy = 0
    for _, label_feature_count in count_of_label_features.items():
        entropy += log_base2(label_feature_count, total_number_of_features)
    return entropy


def calculate_conditional_entropy(dictionary, attribute_name, label_name):
    """
    Calculates conditional entropy for an attribute.

    :param dictionary:
    :param attribute_name:
    :param label_name:
    :return:
    """
    count_dict = get_count_dict_from_dictionary(dictionary, attribute_name, label_name)

    list_of_attribute_features = dictionary[attribute_name]
    set_of_attribute_features = set(list_of_attribute_features)
    total_number_of_features = len(list_of_attribute_features)
    conditional_entropy = 0
    for attribute_feature in set_of_attribute_features:
        conditional_entropy += float(
            count_dict[attribute_feature]['count']) / total_number_of_features * calculate_entropy_given_feature(
            count_dict, attribute_feature)

    return conditional_entropy


def calculate_entropy_given_feature(count_dict, attribute_feature):
    """
    Calculates entropy given an attribute feature

    :param count_dict:
    :param attribute_feature:
    :return: entropy
    """
    entropy = 0
    feature_count = count_dict[attribute_feature]['count']
    label_counts = count_dict[attribute_feature]['label_counts']
    for _, label_feature_count in label_counts.items():
        entropy += log_base2(label_feature_count, feature_count)
    return entropy


def initialize_counting_dictionary(set_of_attribute_features, set_of_label_features):
    """
    Initializes a barebones dictionary to keep count of features for attributes and their corresponding label features.

    :param set_of_attribute_features:
    :param set_of_label_features:
    :return: count dictionary
    """
    count_dict = {}
    for attribute_feature in set_of_attribute_features:
        d = {
            'count': 0,
            'label_counts': {}
        }
        for label_feature in set_of_label_features:
            d['label_counts'][label_feature] = 0
        count_dict[attribute_feature] = d
    return count_dict


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


def dictionary_modifier(dictionary, attribute_name, feature):
    """
    :param dictionary: unmodified dictionry from csv
    :param attribute_name: best attribute returned from unmodified dict
    :param feature: feature1 of attribute. For ex 1
    :return:
    """

    modified_dict = collections.defaultdict(list)
    list_of_att = dictionary[attribute_name]
    for index, v in enumerate(list_of_att):
        for key, value in dictionary.items():
            if key != attribute_name:
                if v == feature:
                    modified_dict[key].append(value[index])
    return modified_dict


def get_count_dict_from_dictionary(dictionary, attribute_name, label_name):
    list_of_label_features = dictionary[label_name]
    list_of_attribute_features = dictionary[attribute_name]

    set_of_label_features = set(list_of_label_features)
    set_of_attribute_features = set(list_of_attribute_features)

    count_dict = initialize_counting_dictionary(set_of_attribute_features, set_of_label_features)

    for index, feature in enumerate(list_of_attribute_features):
        count_dict[feature]['count'] += 1
        count_dict[feature]['label_counts'][list_of_label_features[index]] += 1

    return count_dict


def print_tree(node, depth):
    """

    :param node: root_node
    :param depth: current_depth
    :return:
    """
    if node:
        if isinstance(node, Decision):
            print '|' * depth + node.name
        elif isinstance(node, Node):
            print '|' * depth + node.name
            for attribute_feature, child_node in node.children.items():
                sys.stdout.write(attribute_feature + ' ')
                print_tree(child_node, depth + 1)
        else:
            print node.name


def predict_labels(tree, dictionary, label_name):
    predicted_labels = []
    for index,_ in enumerate(dictionary[label_name]):
        prediction = traverse_tree(index, dictionary, tree, label_name)
        predicted_labels.append(prediction)
    return predicted_labels


def traverse_tree(index, dictionary, tree, label_name):
    #untill we come across a Decision node, keep traversing
    if isinstance(tree, Decision):
        return tree.name
    else:
        list_of_features = dictionary[tree.name]
        attribute_feature = list_of_features[index]
        try:
            child = tree.children[attribute_feature]
            return traverse_tree(index, dictionary, child, label_name)
        except KeyError:
            decision = majority_vote_classifier(dictionary, label_name)
            return decision.name

def error_of_prediction_vs_original(predicted_labels, dictionary, label_name):
    count = 0
    for i, v in enumerate(dictionary[label_name]):
        if v != predicted_labels[i]:
            count+=1
    return float(count)/len(predicted_labels)


def write_prediction_to_output_file(predicted_labels, file_name):
    #output_file_name = '{}.labels'.format(file_name)
    with open(file_name, 'w') as file:
        for predicted_label in predicted_labels:
            file.write("%s\n" %predicted_label)


def write_errors_to_metrics_file(training_error, testing_error, file_name):
    with open(file_name, 'w') as file:
        file.write("error(train) : {}\n error(test) : {}".format(training_error, testing_error))


def main():
    args = parse_command_line()

    max_depth = args.max_depth
    train_dictionary, train_label_name= convert_csv_to_dict(args.train_input)
    test_dictionary, test_label_name= convert_csv_to_dict(args.test_input)

    root_node = construct_decision_tree(train_dictionary, max_depth, 1, train_label_name)
    print_tree(root_node, max_depth)

    predicted_train_labels = predict_labels(root_node, train_dictionary, train_label_name)
    write_prediction_to_output_file(predicted_train_labels, args.train_output)
    training_error = error_of_prediction_vs_original(predicted_train_labels, train_dictionary, train_label_name)

    predicted_test_labels = predict_labels(root_node, test_dictionary, test_label_name)
    write_prediction_to_output_file(predicted_test_labels, args.test_output)
    testing_error = error_of_prediction_vs_original(predicted_test_labels, test_dictionary, test_label_name)

    write_errors_to_metrics_file(training_error, testing_error, args.metrics_output)


if __name__ == '__main__':
    main()
