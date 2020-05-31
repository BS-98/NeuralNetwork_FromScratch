import numpy as np

class Dense:
    
    def __init__(self, nb_inputs, units):
        self.type = "TRAINABLE"
        self.name = "dense"
        
        self.W = np.random.randn(nb_inputs, units)
        self.B = np.random.randn(units)
        
    def forward(self, inputs):
        self.inputs = inputs
        y = np.dot(self.inputs, self.W) + self.B
        
        return y
    
    def backward(self, chain):
        y = np.dot(chain, self.W.T)
        
        return y
    
    def update(self, chain, lr):
        self.W = self.W - lr * np.dot(chain[:, None], self.inputs[None]).T
        self.B = self.B - lr * chain

class RELU:
    
    def __init__(self):
        self.type = "NON-TRAINABLE"
        self.name = "relu"
    
    def forward(self, inputs):
        self.inputs = inputs
        y = self.inputs.copy()
        y[y <= 0] = 0
        
        return y
    
    def backward(self, chain):
        y = self.inputs
        y[y > 0] = 1
        
        return chain * y
    
class Sigmoid:
    
    def __init__(self):
        self.type = "NON-TRAINABLE"
        self.name = "sigmoid"
        
    def forward(self, inputs):
        self.inputs = inputs
        y = 1/(1 + np.exp(-self.inputs))
        
        return y
    
    def backward(self, chain):
        y = chain * (np.exp(-self.inputs)/(1 + np.exp(-self.inputs))**2)
        
        return y
    

class Loss:
    
    def __init__(self):
        self.type = "NON-TRAINABLE"
        self.name = "loss"

        
    def forward(self, y, y_hat):
        self.y_hat = y_hat
        self.y = y
        error = (self.y - self.y_hat)**2
        print(error)
        
        return error
    
    def backward(self, chain):
        y = chain * -2*(self.y - self.y_hat)
        
        return y


nb_input = 2
nb_output = 1
batch_size = 1

X = np.random.randn(30, nb_input)
Y = np.random.randn(30, nb_output)

model = [Dense(nb_input, 3),
         RELU(),
         Dense(3, nb_output),
         Sigmoid(),
         Loss()]   

for _ in range(1000):
    index = np.random.randint(0, 29, batch_size)

    x = X[index].reshape((nb_input))
    y = Y[index].reshape((nb_output))
    
    for layer in model:
        
        if layer.name == "loss":
            x = layer.forward(y, x)
            break
        
        x = layer.forward(x)
        
    chain_back = 1
    chain_update = 1
    
    for layer in model[::-1]:
        chain_back = layer.backward(chain_back)
        
        if layer.type == "TRAINABLE":
            layer.update(chain_update, 0.01)
            
        chain_update = chain_back
        
        
    
    
    
    
    
        