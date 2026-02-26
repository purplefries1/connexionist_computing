import numpy as np

class MLP:

    def __init__(self, NI:int, NH:int, NO:int):
        
        self.num_in:int = NI #number of inputs
        self.num_hidden:int = NH #number of hidden units
        self.num_out:int = NO #number of outputs

        self.w1 = np.zeros((NH, NI)) # weights for input->hidden where each row is a different hidden neuron
        self.w2 = np.zeros((NO, NH)) # weights for hidden->output

        self.dw1 = np.zeros((NH, NI)) # weight changes for input->hidden
        self.dw2 = np.zeros((NO, NH)) # weight changes for hidden->output

        self.z1 = np.zeros(NH) # array containing z, the weighted sums of inputs for the hidden layer
        self.z2 = np.zeros(NO) # same for the output layer
        
        self.h = np.zeros(NH) # outputs of the hidden layer
        self.o = np.zeros(NO) # outputs of the output layer
        

    def randomise(self):
        # Set dw1 and dw2 to arrays full of 0
        self.dw1 = np.zeros((self.num_hidden, self.num_in))
        self.dw2 = np.zeros((self.num_out, self.num_hidden))

        # set w1 and w2 to random values
        self.w1 = np.random.uniform(-0.1, 0.1, (self.num_hidden, self.num_in))
        self.w2 = np.random.uniform(-0.1, 0.1, (self.num_out, self.num_hidden))




    def forward(self, input):

        # lower layer (hidden)
        self.z1 = self.w1 @ input # dot product between weights and inputs
        self.h = self.sigmoid(self.z1)

        # upper layer (output)
        self.z2 = self.w2 @ self.h
        self.o = self.sigmoid(self.z2)



    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        

    def backwards(self, input, target):

        # delta output
        delta_output = ((target - self.o) * (self.o * (1 - self.o))) # delta for output layer with sigmoid

        # dw2 (output layer)
        self.dw2 += np.outer(delta_output, self.h) # weight updates = delta * input (output of the hidden layer)

        # delta hidden
        delta_hidden = (delta_output @ self.w2) * (self.h * (1 - self.h)) # delta for hidden layer with sigmoid

        # dw1 (hidden layer)
        self.dw1 += np.outer(delta_hidden, input)

        # error log
        return 0.5 * np.sum((self.o - target)** 2)
    
    def update_weights(self, learning_rate:float):

        self.w1 += learning_rate * self.dw1
        self.w2 += learning_rate * self.dw2


        self.dw1 = np.zeros((self.num_hidden, self.num_in))
        self.dw2 = np.zeros((self.num_out, self.num_hidden))

    def training(self, max_epochs, training_set, learning_rate, log):
        """training set: [[input, target_output], [input, target_output], ...]"""

        self.randomise()
        e = 0
        n = len(training_set)

        while e < max_epochs:

            error = 0

            for input, target in training_set:
                self.forward(input=input)
                error += self.backwards(input=input, target=target)
            
            self.update_weights(learning_rate=learning_rate)

            checkpoints = self.logs(max_epochs)

            if e in checkpoints: 
                log(f"Error at epoch {e+1} is {error}")
                

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


        