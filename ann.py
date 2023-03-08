import numpy as np


class Network:
    """class to create a neural network"""
    def __init__(self, learning_rate=.3, momentum=.9, number_layers=3, neurons_per_layer=2, number_inputs=9,
                 number_classes=2, function="sigmoid"):
        uid = 0
        self.learning = learning_rate
        self.momentum = momentum
        self.inputs = []
        self.layers = []
        self.nn = []
        self.number_of_classes = number_classes
        self.t = None
        i = 0
        while i < number_inputs + 1:  # create the input layer
            if i == 0:
                self.inputs.append((Bias(uid, 0)))
            else:
                self.inputs.append(Input(uid, 0))
            uid += 1
            i += 1
        i = 1
        while i < number_layers: # create the hidden layers
            j = 0
            layer = []
            while j < neurons_per_layer + 1:
                if j == 0:
                    layer.append((Bias(uid, i)))
                else:
                    layer.append(Neuron(uid, i, function=function))
                uid += 1
                j += 1
            self.layers.append(layer)
            i += 1
        number_of_final_neurons = self.determine_number_output_neurons()
        j = 0
        layer = []
        while j < number_of_final_neurons:  # create the final output layer
            layer.append(Neuron(uid, i, function=function))
            print("added last")
            j += 1
            uid += 1
        self.layers.append(layer)
        self.nn.append(self.inputs)
        self.nn += self.layers
        self.wire_network()

    def get_layer_inputs(self, i):
        """takes input of layer index, returns previous layer"""
        if i == 0:
            layer_inputs = self.inputs
        else:
            layer_inputs = self.layers[i - 1]
        return layer_inputs

    def determine_number_output_neurons(self):
        """used to determine the number of output neurons; 1 for binary classification"""
        if self.number_of_classes == 2:
            final_neurons = 1
        else:
            final_neurons = self.number_of_classes
        return final_neurons

    def wire_network(self, weight=0.1):
        """wires the network, by adding the synapses between layers and intializes the weights"""
        for i, layer in enumerate(self.nn):
            for j, neuron in enumerate(layer):
                try:
                    for endpoint in self.nn[i + 1]:
                        if endpoint.function != "bias":
                            start = self[neuron.uid]
                            end = self[endpoint.uid]
                            this_synapse = Synapse(start, end, weight=weight, state=0)
                            print("creating: %s" % this_synapse)
                            start.outgoing.append(this_synapse)
                            try:
                                end.incoming.append(this_synapse)
                            except AttributeError:
                                print(end.function)
                except IndexError:
                    print("finished wiring neural network")

    def __getitem__(self, item):
        selected = None
        for layer in self.nn:
            for neuron in layer:
                if neuron.uid == item:
                    selected = neuron
                    break
        return selected

    def get_layer(self, layer):
        """takes layer index for input, returns layer"""
        try:
            return self.nn[layer]
        except IndexError:
            print("unable to retrieve layer %s" % layer)

    def load_record(self, record):
        """takes record as input, loads the record into the input neurons and sets target value"""
        for i, input_neuron in enumerate(self.nn[0]):
            if input_neuron.function != "bias":
                input_neuron.load(record[i - 1])
        self.t = record.classification

    def fire(self):
        """single firing of the network; calculates z value for all neurons in network"""
        for i, layer in enumerate(self.nn):
            if i != 0:
                for neuron in layer:
                    if neuron.function != "bias":
                        neuron.calc_z()

    def calculate_error(self, record):
        """calculates error for processed record"""
        i = len(self.nn) - 1
        final_layer = self.get_layer(i)
        for neuron in final_layer:  # sigmoid error function
            neuron.calculate_output_error_sigmoid(self.t)
            record.add_prediction(neuron.z)
        i -= 1
        while i > 0:  # back-propagate error
            this_layer = self.get_layer(i)
            for neuron in this_layer:
                neuron.calculate_backprop_error()
            i -= 1

    def adjust_weights(self):
        """adjust the weights across the network"""
        for layer in self.nn:
            for neuron in layer:
                for synapse in neuron.outgoing:
                    synapse.set_state(self.momentum)
                    synapse.update_weight_with_momentum(self.learning)

    def process_record(self, record, train=True):
        """processes a single record through the network; optional input to train or test the record"""
        self.load_record(record)
        self.fire()
        self.calculate_error(record)
        if train:
            self.adjust_weights()
        return record

    def process_recordset(self, recordset, train=True):
        """processes an entire recordset"""
        for i, record in enumerate(recordset):
            output_record = self.process_record(record, train)
            recordset[i] = output_record
        # print("completed processing records")
        return recordset

    def train_recordset(self, recordset, delay=500):
        """continues to process a recordset until trained, default delay of 500 iterations"""
        i = 0
        error = np.infty
        trained = None
        train = True
        while train:  # keep training
            processed_recordset = self.process_recordset(recordset)
            this_error = processed_recordset.determine_total_error()
            if this_error < error:
                error = this_error
                trained = i
            if i > trained + delay:
                train = False
            i += 1
        return error, trained

    def train_and_test_network(self, training_recordset, test_recordset, delay=500):
        """function to train a training set and then test the trained network using the test set"""
        error, trained = self.train_recordset(training_recordset, delay=delay)
        print("trained network in %s iterations for an error rate of %s" % (trained, error))
        test = self.process_recordset(test_recordset, train=False)
        error = test.determine_total_error()
        print("total error for test record set = %s" % error)
        return error, trained

    def get_weights_for_layer(self, layer_index):
        """function to return the synapse uid and synapse weight for a layer"""
        layer = self.get_layer(layer_index)
        layer_outgoing = []
        for neuron in layer:
            neuron_outgoing = []
            for synapse in neuron.outgoing:
                neuron_outgoing.append([synapse.uid, synapse.weight])
            layer_outgoing.append(neuron_outgoing)
        return layer_outgoing

    def return_input_layer_weights(self):
        """used to get the weights for the input layer, used for graph generation"""
        average_weights = []
        layer = self.get_layer(0)
        for neuron in layer:
            this_weight = []
            for synapse in neuron.outgoing:
                this_weight.append(synapse.weight)
            this_weight = np.array([this_weight])
            this_weight = np.mean(this_weight)
            average_weights.append(this_weight)
        return average_weights

    def reset_network(self, weight=0.1, state=0):
        """resets the network to default settings"""
        for layer in self.nn:
            for neuron in layer:
                for synapse in neuron.outgoing:
                    synapse.weight = weight
                    synapse.state = state


class Neuron:
    """base class for all neurons"""

    def __init__(self, identity, layer, function="sigmoid"):
        self.function = function
        self.uid = str(layer) + "s" + str(identity)
        self.layer = layer
        self.e = None
        self.w = None
        self.x = None
        self.y = None
        self.z = None
        self.incoming = []
        self.outgoing = []

    def __str__(self):
        string = "%s neuron uid: %s in layer %s" % (self.function, self.uid, self.layer)
        return string

    def get_w_x(self):
        """gets inputs for w and x to calculate y"""
        w = []
        x = []
        for synapse in self.incoming:
            w.append(synapse.weight)
            x.append(synapse.start.z)
        self.w = np.array(w)
        self.x = np.array(x)
        return self.w, self.x

    def calc_y(self):
        """calculates y"""
        self.get_w_x()
        self.y = np.dot(self.w.transpose(), self.x)
        return self.y

    def sgn(self):
        """implements the sgn function"""
        if self.y > 0:
            self.z = 1
        else:
            self.z = -1
        return self.z

    def sigmoid(self):
        """implements the sigmoid function"""
        self.z = 1 / (1 + np.e ** -self.y)
        return self.z

    def calc_z(self):
        """calculates z based on neuron function"""
        if self.function == "sigmoid":
            self.calc_y()
            self.sigmoid()
        elif self.function == "sgn":
            self.calc_y()
            self.sgn()
        return self.z

    def calculate_output_error_sigmoid(self, t):
        """calculates error at output neuron"""
        self.e = self.z * ((1 - self.z) * (t - self.z))
        return self.e

    def calculate_backprop_error(self):
        """calculates back propagation error"""
        error = []
        for outgoing_synapse in self.outgoing:
            this_error = outgoing_synapse.end.e * outgoing_synapse.weight * (self.z * (1 - self.z))
            error.append(this_error)
        total_error = sum(error)
        self.e = total_error
        return self.e


class Input(Neuron):
    """class of neurons used in the input layer"""

    def __init__(self, identity, layer):
        super().__init__(identity, layer)
        self.uid = str(layer) + "i" + str(identity)
        self.function = "input"
        self.layer = 0
        self.incoming = None

    def load(self, input_field):
        """loads the neuron with the input field from a record"""
        self.z = input_field
        return self.z


class Bias(Neuron):
    """class of neurons for bias"""

    def __init__(self, identity, layer):
        super().__init__(identity, layer)
        self.function = "bias"
        self.layer = layer
        self.incoming = None
        self.z = 1
        self.uid = str(layer) + "b" + str(identity)

    def calc_z(self):
        return self.z


class Synapse:
    """class for the connection between two neurons"""

    def __init__(self, start, end, weight=0.1, state=0):
        self.start = start
        self.end = end
        self.weight = weight
        self.state = state
        self.uid = str(end.uid) + str("-") + str(start.uid)

    def __str__(self):
        string = "synapse object %s from neuron %s to neuron %s" % (self.uid, self.start.uid, self.end.uid)
        return string

    def set_state(self, momentum=.09):
        """sets / updates the state based on momentum"""
        self.state = ((1 - momentum) * self.end.e * self.start.z) + (momentum * self.state)
        return self.state

    def add_momentum(self, learn_n):
        """adds momentum"""
        weight_change = learn_n * self.state
        return weight_change

    def update_weight_with_momentum(self, learn_n):
        """function to update the weights based on learning rate"""
        self.weight = self.weight + self.add_momentum(learn_n)
        return self.weight


class Record(list):
    """class used as a single record for processing"""

    def __init__(self, row, uid_index, class_index):
        self.exclude = [uid_index, class_index]
        self.classification = row[class_index]
        self.prediction = None
        self.uid = row[uid_index]
        for i, field in enumerate(row):
            if i not in self.exclude:
                self.append(float(field))

    def add_prediction(self, z):
        """adds the predicted value for the record"""
        if z > 0.5:  # modify this to change decision threshold
            self.prediction = 1
        else:
            self.prediction = 0

    def get_prediction(self):
        """returns the prediction"""
        return self.prediction


class RecordSet(list):
    """class to hold the entire set of records"""

    def __init__(self, matrix, uid_index, class_index):
        for row in matrix:
            this_record = Record(row, uid_index, class_index)
            self.append(this_record)
        self.total_error = None
        self.uid_index = uid_index
        self.class_index = class_index

    def determine_total_error(self):
        """determines cross class entropy; currently based on binary classes"""
        incorrect = 0
        for record in self:
            if record.classification != record.get_prediction():
                incorrect += 1
        error = incorrect / len(self)
        self.total_error = error
        return self.total_error
