import numpy as np

class MLP:

    def __init__(self, NI:int, NH:int, NO:int, hidden_type:str, output_type:str):
        
        self.num_in:int = NI #number of inputs
        self.num_hidden:int = NH #number of hidden units
        self.num_out:int = NO #number of outputs

        self.w1 = np.zeros((NH, NI)) # weights for input->hidden where each row is a different hidden neuron
        self.w2 = np.zeros((NO, NH)) # weights for hidden->output

        self.dw1 = np.zeros((NH, NI)) # weight changes for input->hidden
        self.dw2 = np.zeros((NO, NH)) # weight changes for hidden->output

        self.b1 = np.zeros(NH) # biases for hidden layer
        self.b2 = np.zeros(NO) # biases for output layer

        self.db1 = np.zeros(NH) # weight changes for b1
        self.db2 = np.zeros(NO) # weight changes for b2

        self.z1 = np.zeros(NH) # array containing z, the weighted sums of inputs for the hidden layer
        self.z2 = np.zeros(NO) # same for the output layer
        
        self.h = np.zeros(NH) # outputs of the hidden layer
        self.o = np.zeros(NO) # outputs of the output layer

        self.hidden_type = hidden_type # S (sigmoid), T (tanh), L (linear)
        self.output_type = output_type
        

    def randomise(self):
        # Set dw1 and dw2 to arrays full of 0
        self.dw1 = np.zeros((self.num_hidden, self.num_in))
        self.dw2 = np.zeros((self.num_out, self.num_hidden))
        self.db1 = np.zeros(self.num_hidden)
        self.db2 = np.zeros(self.num_out)


        # set w1 and w2 to random values
        limit1 = 1 / np.sqrt(self.num_in)
        self.w1 = np.random.uniform(-limit1, limit1, (self.num_hidden, self.num_in))

        limit2 = 1 / np.sqrt(self.num_hidden)
        self.w2 = np.random.uniform(-limit2, limit2, (self.num_out, self.num_hidden))

        #self.w1 = np.random.uniform(-1, 1, (self.num_hidden, self.num_in))

        #self.w2 = np.random.uniform(-1, 1, (self.num_out, self.num_hidden))






    def forward(self, input):

        # lower layer (hidden)
        self.z1 = self.w1 @ input + self.b1 # dot product between weights and inputs + bias

        if self.hidden_type == "S":
            self.h = self.sigmoid(self.z1)

        elif self.hidden_type == "T":
            self.h = self.tanh(self.z1)

        else:
            raise ValueError("Unsupported hidden layer activation type")

        # upper layer (output)
        self.z2 = self.w2 @ self.h + self.b2
        if self.output_type == "S":
            self.o = self.sigmoid(self.z2)
        
        elif self.output_type == "T":
            self.o = self.tanh(self.z2)

        elif self.output_type == "L":   # Linear
            self.o = self.z2            # pas d'activation

            
        else:
            raise ValueError("Unsupported output layer activation type")



    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def tanh(self, z):
        return np.tanh(z)
    
        

    def backwards(self, input, target):

        # delta output
        if self.output_type == "S":
            delta_output = ((target - self.o) * (self.o * (1 - self.o))) # delta for output layer with sigmoid
        
        elif self.output_type == "T":
            delta_output = ((target - self.o) * (1 - self.o ** 2))

        elif self.output_type == "L":
            delta_output = (target - self.o) 



        else:
            raise ValueError("Unsupported output layer activation type")

        # dw2 (output layer)
        self.dw2 += np.outer(delta_output, self.h) # weight updates = delta * input (output of the hidden layer)

        # biases for output neurons
        self.db2 += delta_output

        # delta hidden
        if self.hidden_type == "S":
            delta_hidden = (delta_output @ self.w2) * (self.h * (1 - self.h)) # delta for hidden layer with sigmoid

        elif self.hidden_type == "T":
            delta_hidden = (delta_output @ self.w2) * (1 - self.h ** 2)

        else:
            raise ValueError("Unsupported hidden layer activation type")

        # dw1 (hidden layer)
        self.dw1 += np.outer(delta_hidden, input)

        # biasses for hidden layer
        self.db1 += delta_hidden 

        # error log
        return 0.5 * np.sum((self.o - target)** 2)
    
    def update_weights(self, learning_rate:float):

        self.w1 += learning_rate * self.dw1
        self.w2 += learning_rate * self.dw2

        # update biases
        self.b1 += learning_rate * self.db1
        self.b2 += learning_rate * self.db2


        self.dw1 = np.zeros((self.num_hidden, self.num_in))
        self.dw2 = np.zeros((self.num_out, self.num_hidden))
        self.db1 = np.zeros(self.num_hidden)
        self.db2 = np.zeros(self.num_out)


    def training(self, max_epochs, training_set, learning_rate, log, testing_set=None): # Testing set is used for measuring overfitting with the SIN problem
        """training set: [[input, target_output], [input, target_output], ...]"""

        self.randomise()
        e = 0
        n = len(training_set)

        while e < max_epochs:

            error = 0

            # training (forward + backwards for each example)
            for input, target in training_set:

                self.forward(input=input)
                error += self.backwards(input=input, target=target)


            # mean of weight updates
            self.dw1 /= n
            self.dw2 /= n
            self.db1 /= n
            self.db2 /= n
            
            self.update_weights(learning_rate=learning_rate)

            checkpoints = self.logs(max_epochs)

            if e in checkpoints: 

                log(f"Error at epoch {e+1} is {error}")
                test_error = 0.0

                if testing_set is not None: # for the sin problem, we measure test error to monitor overfitting
                    for x, t in testing_set:
                        self.forward(x)
                        out = float(self.o[0])
                        target = float(t[0])
                        test_error += 0.5 * (out - target)**2

                    test_error /= len(testing_set)
                    train_mean = error / n
                    log(f"Train mean error at (after) epoch {e+1}: {train_mean}")


                    log(f"Test error at (after) epoch {e+1}: {test_error}")
                    log("----------")

            

            e += 1

    def logs(self, max_epochs):
        """
        Return a set of epoch indices where the training error should be printed
            - If max_epochs < 10 -> print the error for every epoch
            - Otherwise -> print 10 uniformly spaced checkpoints
                (0%, 10%, 20%, ..., 100%).
        """

        last = max_epochs - 1

        if max_epochs < 10:
            return set(range(max_epochs))

        checkpoints = set()
        for i in range(11):
            checkpoints.add((last * i) // 10)

        checkpoints.add(last)

        return checkpoints


        