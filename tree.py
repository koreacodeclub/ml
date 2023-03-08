import copy
import numpy as np
import information as info
import tree
import matplotlib.pyplot as plt


class Tree:
    """base class for the tree dataset"""

    def __init__(self, matrix, class_index=6):
        self.root = self
        self.depth = 0
        self.class_index = class_index  # column that contains the classification
        self.matrix = matrix
        self.children = []
        self.tree_array = []  # contains all nodes in the tree
        self.rule = None  # returns the decision path
        self.parent = None
        self.prediction = None  # returns the majority predictor at the node
        self.unique_id = 0  # unique id for the node
        max_column, max_gain, column_gains = info.find_max_gain(self.matrix, self.class_index)
        self.decision_index = max_column
        self.decision_column = info.get_column(self.matrix, self.decision_index)
        self.decision_gain = max_gain
        self.procreate()  # function to recursively create the tree
        self.assign_unique_ids()  # assigns unique ids for the nodes in tree_array

    def create_child(self, selected_attribute):
        """creates the specified child, takes a selected attribute as input"""
        s, r, c, a = info.select_attribute(self.matrix, self.decision_index, selected_attribute)
        child = Node(s, self, c, a)
        self.children.append(child)
        self.root.tree_array.append(child)

    def procreate(self):
        """recursive function to continue to create children nodes until the tree bottoms out"""
        unique, uni_numb = info.count_values(self.decision_column)
        for attribute in unique:
            self.create_child(attribute)

    def select_unique_node(self, unique_id):
        """used to select a node by unique id from the tree array"""
        for index, node in enumerate(self.tree_array):
            if int(node.unique_id) == int(unique_id):
                selected = node
                selected_index = index
        return selected_index, selected

    def trial_prune(self, unique_id):
        """function to create a trail prune, takes a node by unique id as input
        returns a pruned tree with node removed from tree"""
        test_tree = copy.deepcopy(self)
        i, node_to_prune = test_tree.select_unique_node(unique_id)
        node_to_prune.parent.remove_child(node_to_prune.unique_id)
        return test_tree

    def prune(self, prune_set):
        """function to fully prune a tree. Takes a pruning set as input
        determines nodes to prune (based on error rate) and removes them from the tree"""
        to_prune = []
        for node in self.tree_array:
            as_is_classified = tree.classify_set(prune_set, self)
            as_is_error = tree.determine_total_error(as_is_classified)
            as_pruned = self.trial_prune(node.unique_id)
            pruned_test = tree.classify_set(prune_set, as_pruned)
            as_pruned_error = tree.determine_total_error(pruned_test)
            if as_pruned_error < as_is_error:
                to_prune.append(node)
        if len(to_prune) > 0:
            for node_to_prune in to_prune:
                print("removing: %s" % node_to_prune.unique_id)
                node_to_prune.parent.remove_child(node_to_prune.unique_id)
            to_prune = []
            self.prune(prune_set)

    def remove_child(self, unique_id):
        """removes a child from a node by unique id"""
        selected_index = None
        selected = None
        for index, node in enumerate(self.children):
            if int(node.unique_id) == int(unique_id):
                selected = node
                selected_index = index
        if selected_index is not None:
            self.children.pop(selected_index)
        return selected_index, selected

    def assign_unique_ids(self):
        """assigns unique ids for all nodes in tree array"""
        for i, node in enumerate(self.tree_array):
            self.tree_array[i].unique_id = i + 1


class Node(Tree):
    """node class. inherits from tree"""

    def __init__(self, matrix, parent, column, attribute):
        # print("column:  %s attribute: %s" % (column, attribute))
        self.depth = parent.depth + 1
        self.matrix = matrix
        self.parent = parent
        self.root = parent.root
        self.unique_id = None
        self.class_index = parent.class_index
        self.children = []
        self.tree_array = self.root.tree_array
        self.rule = [column, attribute]
        max_column, max_gain, column_gains = info.find_max_gain(self.matrix, self.class_index)
        self.decision_index = max_column
        self.decision_column = info.get_column(self.matrix, self.decision_index)
        self.decision_gain = max_gain
        self.majority_classifier()  # determine prediction at node
        if self.decision_gain > 0:
            self.procreate()
        else:
            self.path = get_decision_path(self, path=[])

    def print_rule(self, indent):
        """returns the decision path rule for the node. indent is used to indent a full tree"""
        rule = self.rule
        ind = indent * 3
        ind = " " * ind
        try:
            print("%s depth: %s column: %s attribute: %s" % (ind, self.depth, rule[0], rule[1]))
            print("%s prediction: %s" % (ind, self.prediction))
        except IndexError:
            print("rule of out range")
            print(rule)

    def majority_classifier(self):
        """function used to determine the prediciton at a node, based on majority voting"""
        class_counts = [0, 0, 0, 0]
        class_column = info.get_column(self.matrix, self.class_index)
        for row in class_column:
            class_counts[int(row)] = class_counts[int(row)] + 1
        max_count = -np.infty
        max_class = None
        for i, v in enumerate(class_counts):
            if v > max_count:
                max_count = v
                max_class = i
        self.prediction = int(max_class)


def print_tree(tree, indent=0):
    """prints the tree for visual display"""
    if indent == 0:
        print("Creating Tree...")
    indent += 1
    if len(tree.children) > 0:
        for child in tree.children:
            child.print_rule(indent)
            print_tree(child, indent)
    else:
        ind = indent * 3
        ind = " " * ind
        print("%s decision path = %s" % (ind, tree.path))


def get_decision_path(node, path=[]):
    """takes a node as input, returns the decision path"""
    path.append(node.rule)
    if node.parent is not None:
        get_decision_path(node.parent, path=path)
    return path


def seek_bottom(record, this_tree, class_index=6):
    """sends a single record down a tree to find its appropriate predicted classification
    takes a record, tree, and class_index as input, returns the bottom node"""
    if len(this_tree.children) == 0:
        bottom = this_tree
    else:
        evaluate_attribute = record[this_tree.decision_index]
        for child in this_tree.children:
            if int(child.rule[1]) == int(evaluate_attribute):
                bottom = seek_bottom(record, child)
    if 'bottom' not in locals():  # used for not situations
        # print("false bottom at: %s" % tree.depth)
        attributes = ""
        for child in this_tree.children:
            attributes = attributes + " " + str(child.rule[1])
        # print("evaluation attribute: %s is not%s" % (evaluate_attribute, attributes))
        bottom = this_tree
    return bottom


def classify_record(record, this_tree, class_index=6):
    """used to classify as single record, finds bottom and classifies it as such."""
    copied = copy.deepcopy(record)
    bottom = seek_bottom(copied, this_tree)
    if bottom.prediction is None:  # used to fix error in some nodes not being classified
        quick_fix_classifier(bottom)
    prediction = bottom.prediction
    copied.append(prediction)
    if int(copied[class_index]) == int(copied[class_index + 1]):
        correct = 1
    else:
        correct = 0
    copied.append(correct)
    return copied


def classify_set(matrix, tree, class_index=6):
    """takes a matrix, and tree as input. returns a matrix with a prediction column appended"""
    processes_matrix = []
    for record in matrix:
        processed_record = classify_record(record, tree, class_index)
        processes_matrix.append(processed_record)
    return processes_matrix


def determine_total_error(matrix, correct_index=8):
    """used to determine the total error in a matrix, returns the error"""
    total = 0
    incorrect = 0
    for row in matrix:
        if row[correct_index] == 0:
            incorrect += 1
        total += 1
    error = incorrect / total
    return error


def average_errors(original_errors, pruned_errors):
    """used to average errors over folds"""
    k = len(original_errors)
    original_error = sum(original_errors) / k
    pruned_error = sum(pruned_errors) / k
    print(original_error)
    print(pruned_error)
    return original_error, pruned_error


def average_error_by_class(original_errors_by_class, pruned_errors_by_class, numb_classes=4):
    """averages errors by class over folds"""
    i = 0
    average_errors_by_class = []
    average_pruned_error_by_class = []
    while i < numb_classes:
        original = select_column_average(original_errors_by_class, i)
        pruned = select_column_average(pruned_errors_by_class, i)
        average_errors_by_class.append(original)
        average_pruned_error_by_class.append(pruned)
        i += 1
    return average_errors_by_class, average_pruned_error_by_class


def select_column_average(matrix, column):
    """selects a column from a matrix and returns the average"""
    return_column = []
    for record in matrix:
        return_column.append(float(record[column]))
    k = len(return_column)
    average = sum(return_column) / k
    return average


def graph_average_error(original_error, pruned_error, title, y_lab="Error", x_lab="Results"):
    """graphs the average error"""
    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    plt.bar(1, original_error, color="red", edgecolor="black", label="original")
    plt.bar(2, pruned_error, color="blue", edgecolor="black", label="pruned")
    plt.legend()
    plt.show()


def graph_average_error_by_class(by_class_original, by_class_pruned):
    """graphs average error by a single class"""
    for i, c in enumerate(by_class_original):
        title = "Class %s: Average Errors" % i
        x_lab = "Class %s" % i
        graph_average_error(by_class_original[i], by_class_pruned[i], title, x_lab=x_lab)


def full_graph_by_class(by_class_original, by_class_pruned, title="Figure 2: By Class Error",
                        y_lab="Error", x_lab="Classes"):
    """graphs all the classes by error, see figure 2"""
    plt.title(title)
    plt.ylabel(y_lab)
    plt.xlabel(x_lab)
    for i, c in enumerate(by_class_original):
        plt.bar(i, by_class_original[i], color="red", edgecolor="black", label="original", width=.5, align="edge")
        plt.bar(i, by_class_pruned[i], color="blue", edgecolor="black", label="pruned", width=-.5, align="edge")
    plt.legend()
    plt.show()


def quick_fix_classifier(false_bottom):
    """used to fix an issue present in tree after pruning, when a node does not have a prediction"""
    class_counts = [0, 0, 0, 0]
    class_column = info.get_column(false_bottom.matrix, false_bottom.class_index)
    for row in class_column:
        class_counts[int(row)] = class_counts[int(row)] + 1
    max_count = -np.infty
    max_class = None
    for i, v in enumerate(class_counts):
        if v > max_count:
            max_count = v
            max_class = i
    false_bottom.prediction = int(max_class)


def separate_results_by_class(processed_records, number_of_classes=4, class_index=6):
    """used to separate the processed results by class. used to prep data to determine by class error."""
    results_by_class = []
    i = 0
    while i < number_of_classes:
        results_by_class.append([])
        i += 1
    for record in processed_records:
        record_class = int(record[class_index])
        results_by_class[record_class].append(record)
    return results_by_class


def do_work_project3(training, test, prune_test):
    """function designed for project 3, takes the training, test, and pruning datasets
    as input, classifies the datasets, determines the error rate, and provides a graphical result"""
    original_results = []
    pruned_results = []
    pruned_trees = []
    by_class_results = []
    by_class_pruned_results = []
    for i, train in enumerate(training):
        ot = tree.Tree(training[i])
        pt = copy.deepcopy(ot)  # creates a copy of the original tree to prune
        pt.prune(prune_test)  # prunes the tree according to the test set
        processed_record = tree.classify_set(test[i], ot)  # determines the classification of records in set
        processed_pruned_record = tree.classify_set(test[i], pt)
        o_error = tree.determine_total_error(processed_record)  # determines the error rate in the set
        p_error = tree.determine_total_error(processed_pruned_record)
        original_results.append(o_error)
        pruned_results.append(p_error)
        pruned_trees.append(pt)
        by_class = tree.separate_results_by_class(processed_record)  # separates the results by class
        by_class_pruned = tree.separate_results_by_class(processed_pruned_record)
        fold_by_class_results = []
        fold_by_class_pruned_results = []
        for c, this_class in enumerate(by_class):  # iterates to produce the error per class per fold
            o_class_e = tree.determine_total_error(this_class)
            p_class_e = tree.determine_total_error(by_class_pruned[c])
            fold_by_class_results.append(o_class_e)
            fold_by_class_pruned_results.append(p_class_e)
        by_class_results.append(fold_by_class_results)
        by_class_pruned_results.append(fold_by_class_pruned_results)
    for i in range(len(pruned_trees)):
        print("fold %s original error = %s, pruned error = %s" % (i, original_results[i], pruned_results[i]))
    by_class_original_errors, by_class_pruned_errors = tree.average_error_by_class(by_class_results,
                                                                                   by_class_pruned_results)  # averages
    # error by class
    original_error, pruned_error = tree.average_errors(original_results, pruned_results)
    tree.graph_average_error(original_error, pruned_error, "Figure 1: Average Errors")  # displays figure 1
    tree.full_graph_by_class(by_class_original_errors, by_class_pruned_errors)  # displays figure 2

# do_work_project3(train_sets, test_sets, prune_set)"""
