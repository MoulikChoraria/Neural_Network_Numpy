import random
import numpy as np
import os
from datetime import datetime


class Layer:

    def __init__(self, no_of_neurons, act_fxn="Relu"):
        
        self.size = no_of_neurons
        self.activation(act_fxn)
        
    def activation(self, act="Relu"):
        self.activation = getattr(self, act)
        self.act_fxn = act
        self.gradient_fxn = getattr(self, self.act_fxn + "_grad")

    def activate_layer(self, activations):
        self.activations = activations
        self.output = self.activation(self.activations)
    
    def Relu(self, activations):
        #self.activations = activations
        x = np.where(activations > 0, activations, 0.01*activations)
        return x
        
    
    def Sigmoid(self, activations):
        e = np.exp(-activations)
        return (1/(1+e))
        
    def Relu_grad(self):
        #self.activations = activations
        x = np.zeros(self.activations.shape)
        x = np.where(self.activations > 0, 1, 0.01)
        return x

        
    def Sigmoid_grad(self):
        #self.activations = activations
        #print(self.output, "This is output")
        #print(self.output * (1 - self.output), "gradient")
        return self.output * (1 - self.output)
    
    def ret_layer_grad(self):
        return self.gradient_fxn()
        

    def ret_output(self):
        return self.output

    def ret_activations(self):
        return self.activations

    def ret_grad(self):
        return self.gradient
    
    
    class NeuralNetwork:

    def __init__(self, size_network, input_dim, loss="BCE", batch_size=1):

        self.layers = []
        self.num_layers = 0
        self.size = size_network
        
        self.input_dim = input_dim
        

        for i in size_network:
            self.layers.append(Layer(i))
            self.num_layers = self.num_layers + 1
        
        self.set_loss(loss)
        self.initialize_weights()
        self.batch_size = batch_size
        
    def set_loss(self, loss):
        self.cost = getattr(self, loss)
        self.loss = loss
        self.grad_loss = getattr(self, loss+"_grad")
        
    def ret_loss(self, x, y):
        cost = self.cost(x, y)
        return cost
    
    def ret_grad_loss(self, x, y):
        grad = self.grad_loss(x, y)
        return grad
        

    def add_layer(self, size, act_fxn="Relu"): #input_dim = None):
        self.num_layers = self.num_layers + 1
        self.size.append(size)
        
        #if(input_dim != None):
        #    self.input_dim = input_dim
        self.layers.append(Layer(size, act_fxn))
        self.biases.append(np.random.rand(size, 1))
        self.weights.append(np.random.rand(size, self.layers[-2].size))
        
    def initialize_weights(self):

        self.biases = [np.random.rand(y, 1) for y in self.size[:]]
        self.weights = [np.random.rand(self.layers[0].size, self.input_dim)]
        for x, y in zip(self.layers[1:], self.layers[:-1]):
            self.weights.append(np.random.rand(x.size, y.size)) 
        

    def forward_pass(self, input_x):

        next_input = input_x
        #print(input_x.shape)
        for layer, weight, bias in zip(self.layers, self.weights, self.biases):

            activations = (next_input@weight.T) + bias.T
            layer.activate_layer(activations)
            next_input = layer.ret_output()
            #print(next_input, "layers")
            next_input = np.array(next_input)
        return next_input

    def back_prop(self, x, y):
        delta_bias = [np.zeros(np.shape(b)) for b in self.biases]
        delta_weight = [np.zeros(np.shape(w)) for w in self.weights]
        
        grad = self.ret_grad_loss(self.layers[-1].output, y)
        #print(loss)
        delta = np.array(grad)
        #print("delta_shape_1",delta.shape)
        
        for l in range(1, self.num_layers+1):
            
            layer_gradient = self.layers[-l].ret_layer_grad()
            #print(y, "This is label")
            #print(layer_gradient.shape)
            delta = delta * layer_gradient
            
            #print("delta_shape_1",delta.shape)
           
            delta_bias[-l] = np.sum(delta.T, axis = 1)
            #print(delta_bias[-l].shape, 'bias_shape')
            
            #print(delta_bias[-l].shape)
            
            if(l == self.num_layers):
                delta_weight[-l] = (delta.T @ x)
                
            else:
                delta_weight[-l] = (delta.T @ self.layers[-l - 1].output)
            
            delta = delta @ self.weights[-l]
            #print(delta.shape, "delta_shape")
            #print(delta_weight[-l].shape, "weight_shape")
            
        return delta_bias, delta_weight

    def split_data(self, x, y, ratio=0.75, seed=1):

        # set seed
        random.seed(datetime.now())
        perm = np.random.shuffle(np.arange(x.shape[0]))

        x_shuffled = x[perm]
        y_shuffled = y[perm]

        x_shuffled = np.squeeze(x_shuffled)
        y_shuffled = np.squeeze(y_shuffled)

        x_train = x_shuffled[0:int(x.shape[0]*ratio)]
        y_train = y_shuffled[0:int(x.shape[0]*ratio)]

        x_test = x_shuffled[int(x.shape[0]*ratio):]
        y_test = y_shuffled[int(x.shape[0]*ratio):]

        return x_train, y_train, x_test, y_test
        
    
    def SGD_training(self, x, y , batch_size, eta, epochs=10):

        
        self.batch_size = batch_size
        cost = []
        b_w = []
        b_b = []
        best_TA = 0
        num_batch = 0

        for j in range(epochs):
            data = self.split_data(x, y)
            x_train = np.array(data[0])
            y_train = np.array(data[1])
            n = y_train.shape[0]
            batches = [[x_train[k*batch_size:(k+1)*batch_size], y_train[k*batch_size:(k+1)*batch_size]] for k in range(0, int(n/batch_size))]
            for batch in batches:
                self.update_batch(batch, eta)
                num_batch = num_batch + 1
                
            #print(num_batch, num_batch * batch_size, n)
            train_accuracy = 0.
            batches = [[x_train[k*batch_size:(k+1)*batch_size], y_train[k*batch_size:(k+1)*batch_size]] for k in range(0, int(n/batch_size))]
            num_batch = 0

            for batch in batches:
                pass_thru = batch[0]
                if(self.batch_size == 1):
                    pass_thru = batch[0][np.newaxis, :]   
                xme = self.forward_pass(pass_thru)
                xme = np.where(xme > 0.5, 1, 0)
                #print(xme, batch[1])
                batch[1]=batch[1][:,np.newaxis] 
                train_accuracy = train_accuracy + np.sum(np.where(xme==batch[1], 1, 0))/batch_size
                num_batch = num_batch + 1
            
            train_accuracy=train_accuracy/num_batch
            
            #if(j % 500 == 0):
            #print(self.weights, self.biases)
                
            #epoch_cost = 0
            test_accuracy = 0.
            num_batch = 0
            x_test, y_test = np.array(data[2]), np.array(data[3])
            n = y_test.shape[0]
            batches = [[x_test[k*batch_size:(k+1)*batch_size], y_test[k*batch_size:(k+1)*batch_size]] for k in range(0, int(n/batch_size))]
            for batch in batches:
                pass_thru = batch[0]
                num_batch = num_batch + 1
                if(self.batch_size == 1):
                    pass_thru = batch[0][np.newaxis, :]
                
                xme = self.forward_pass(pass_thru)
                xme = np.where(xme > 0.5, 1, 0)
                batch[1]=batch[1][:,np.newaxis] 
                test_accuracy = test_accuracy + np.sum(np.where(xme==batch[1], 1, 0))/batch_size
            
            test_accuracy=test_accuracy/num_batch
            if(best_TA < test_accuracy):
                b_w.append(self.weights)
                b_b.append(self.biases)
                best_TA = test_accuracy
                
            #epoch_cost=epoch_cost/n
            #cost.append(epoch_cost)
            if(j % 1 == 0):
                print ("Epoch %s: Complete" % j)
                print("Training Accuracy: %f  " % train_accuracy)
                print("Testing Accuracy: %f  " % test_accuracy)
            
        print("Training_complete")

        return cost, b_w, b_b

    def update_batch(self, batch, eta):

        delta_bias_batch = [np.zeros(np.shape(b)) for b in self.biases]
        delta_weight_batch = [np.zeros(np.shape(w)) for w in self.weights]
        #print("input_shape",batch[0].shape, batch[1].shape)
        x = batch[0]
        y = batch[1]
        
        #print(x.shape)
        self.forward_pass(x)
        delta_bias_backprop, delta_weight_backprop = self.back_prop(x, y)
        delta_bias_batch = [(nb.T + dnb.T).T for nb, dnb in zip(delta_bias_batch, delta_bias_backprop)]
        delta_weight_batch = [nw + dnw for nw, dnw in zip(delta_weight_batch, delta_weight_backprop)]

        self.weights = [w - eta * nw
                        for w, nw in zip(self.weights, delta_weight_batch)]
        self.biases = [b - eta * nb
                       for b, nb in zip(self.biases, delta_bias_batch)]

    def MSE(self, x, y):
        cost = 0.0
        a = np.array(x)
        if (a.ndim > 1):
            cost += np.sum((a-y)*(a-y))/(2*a.shape[0])
        else:
            cost += np.sum((a-y)*(a-y))/(2)
        return cost
    
    def MSE_grad(self, x, y):
        grad = 0.0
        a = np.array(x)
        grad += (a-y)/(a.shape[0])
        return grad
    
    def BCE(self, y_hat, y):
        y_hat = np.array(y_hat)
        cost = 0.0
        print(-np.log(y_hat + 0.000001)[y == 1].shape)
        cost = np.sum(-np.log(y_hat + 0.000001)[y == 1])
        cost = cost + np.sum(-np.log(1 - y_hat + 0.000001)[y == 0])
        if(y_hat.ndim > 1):
            cost = cost/y_hat.shape[0]
        return cost
                       
    def BCE_grad(self, x, y):
        grad = 0.0
        a = np.array(x)
        y = np.array(y)
        if(y.ndim == 1):
            y = y[:, np.newaxis]
        #print("a", a.shape)
        grad += (a-y)/(a*(1-a) + 0.00001)
        if(a.ndim > 1):
            grad = grad/a.shape[0]
        #print(grad.shape)
        return grad
    
  