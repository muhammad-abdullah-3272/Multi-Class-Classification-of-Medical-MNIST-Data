#------------------------------------------------------------------------------
# Importing Necessary libraires  
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

#------------------------------------------------------------------------------
# Importing Datasets
#------------------------------------------------------------------------------
from Data_Preprocessing import Data



# Instantiate the class Data
data = Data()

# Load the Medical MNIST Data
X_train, Y_train, X_val, Y_val, X_test, Y_test = data.Load_Data()

# Scale the Data
X_train, X_val, X_test = data.Scale(X_train, X_val, X_test)

# == Plot a sample Image (optional) == #
# data.plot_sample(X_train, Y_train, 0)

#==============================================================================
# Initializing Class for Dense layers in Neural Network
#==============================================================================
class Dense_Layer:
    #--------------------------------------------------------------------------
    # Layer Initialization
    #--------------------------------------------------------------------------
    def __init__(self, n_inputs, n_neurons, lamda=0):
        '''
        Parameters
        ----------
        n_inputs : TYPE Integer
            DESCRIPTION.
        n_neurons : TYPE Integer
            DESCRIPTION.
        lamda : TYPE Integer '0', optional
            DESCRIPTION. This function creates Network's layers and initializes wieghts and biases

        Returns
        -------
        None.

        '''
        # Initializing weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        # Initializing regularization parameter (lambda)
        self.lamda = lamda    
        
    #--------------------------------------------------------------------------
    # Forward Propagation
    #--------------------------------------------------------------------------
    def Forward(self, xTrain, xVal):
        '''
        Parameters
        ----------
        xTrain : TYPE Numpy Array
        xVal : TYPE Numpy Array
            DESCRIPTION: This function takes training and validation datasset for forward pass

        Returns
        -------
        None.

        '''
        # Save Input Datasets values
        self.xTrain = xTrain
        self.xVal = xVal
        
        # Calculate outputs from input datasets, weights and biases
        self.output = np.dot(xTrain, self.weights) + self.biases
        self.val_output = np.dot(xVal, self.weights) + self.biases
        
    #--------------------------------------------------------------------------   
    # Backward Propagation
    #--------------------------------------------------------------------------
    def Backward(self, d_values):
        '''
        Parameters
        ----------
        d_values : TYPE Numpy Array
            DESCRIPTION: This function performs backprop using chain rule

        Returns
        -------
        None.

        '''
        # Gradients on parameters lambdas and biases
        self.d_weights = np.dot(self.xTrain.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
        
        # Gradient on regularization, L2 Regularization on weights
        if self.lamda > 0:
            self.d_weights = self.d_weights + 2 * self.lamda * self.weights
            
        # Gradient on values
        self.d_inputs = np.dot(d_values, self.weights.T)
        

#==============================================================================
# Softmax Activation
#==============================================================================
class Softmax_Activation:
    #--------------------------------------------------------------------------
    # Forward Pass
    #--------------------------------------------------------------------------
    def Forward(self, xTrain, xVal):
        '''
        Parameters
        ----------
        xTrain : TYPE Numpy Array
        xVal : TYPE Numpy Array
            DESCRIPTION: This function performs Sigmoid activation on training datasset 
            while forward propagation

        Returns
        -------
        None.
        '''
        # Remember Input Values
        self.xTrain = xTrain
        self.xVal = xVal
        
        # Get unnormalized probabilities
        train_exp_values = np.exp(xTrain - np.max(xTrain, axis=1, keepdims=True))
        val_exp_values = np.exp(xVal - np.max(xVal, axis=1, keepdims=True))
        
        # Normalize the above probabilities
        train_probabilities = train_exp_values / (np.sum(train_exp_values, axis=1, keepdims=True))
        val_probabilities = val_exp_values / (np.sum(val_exp_values, axis=1, keepdims=True))

        # Estimate the probabilities as output
        self.output = train_probabilities
        self.val_output = val_probabilities
        
        
    #--------------------------------------------------------------------------
    # Backward Propagation
    #--------------------------------------------------------------------------
    def Backward(self, d_values):
        '''
        Parameters
        ----------
        d_values : TYPE Numpy Array
            DESCRIPTION: This function performs backprop on Sigmoid activation

        Returns
        -------
        None.
        '''
        
        # Create un-initialized array
        self.d_inputs = np.empty_like(d_values)
        
        # Enumerate calculated outputs and gradientsoutputs and gradients 
        for index, (single_output, single_d_values) in enumerate(zip(self.output, d_values)):
            
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            
            # Calculate Jacobian Matrix of the Output
            jacob_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            # Calculate sample-wise Gradient and Add it to the array of sample Gradients
            self.d_inputs[index] = np.dot(jacob_matrix, single_d_values)
        
#==============================================================================        
# Gradient Descent
#==============================================================================
class GD_Optimizer:
    #--------------------------------------------------------------------------
    # Initialize Optimizer - Set settings
    # Learning rate (alpha) is set as 0.7
    #--------------------------------------------------------------------------
    def __init__(self, alpha=0.1):
        '''
        Parameters
        ----------
        alpha : TYPE Int, optional
            DESCRIPTION. The default is 0.7.
        decay : TYPE Int, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        '''
        self.alpha = alpha
        

    #--------------------------------------------------------------------------    
    # Update Parameters
    #--------------------------------------------------------------------------
    def Update_Params(self, layer):
        '''
        Parameters
        ----------
        layer : TYPE Numpy Array
            DESCRIPTION: This function updates the parameters in each corresponding layer

        Returns
        -------
        None.

        '''
        layer.weights = layer.weights - self.alpha * layer.d_weights
        layer.biases = layer.biases - self.alpha * layer.d_biases
    

#==============================================================================
# Class to Calculate Common Loss
#==============================================================================
class Loss:
    #--------------------------------------------------------------------------
    # Regularization Loss Calculation
    #--------------------------------------------------------------------------
    def Loss_Regularization(self, layer):
        '''
        Parameters
        ----------
        layer : TYPE Numpy Array
            DESCRIPTION.Regularization loss is calculated in each layer

        Returns
        -------
        loss_regularization : TYPE Int
            DESCRIPTION. Scalar value is returned as regularization loss

        '''
        loss_regularization = 0                     # 0 as default
        
        # L2 Regularization - weights, calculate only when factor greater than 0
        if layer.lamda > 0:         
            loss_regularization = loss_regularization + layer.lamda * np.sum(layer.weights * layer.weights)
        return loss_regularization
    
    #--------------------------------------------------------------------------
    # Calculate the data and regularization loss for given model output and actual labels
    #--------------------------------------------------------------------------
    def Calculate(self, predicted, labels):
        '''
        Parameters
        ----------
        predicted : TYPE Numpy Array
            DESCRIPTION. Model Output (Predicted Values)
        labels : TYPE Numpy Array
            DESCRIPTION. Actual labels

        Returns
        -------
        data_loss : TYPE Int
            DESCRIPTION. Mean Loss

        '''
        # Calculate Sample Losses
        sample_losses = self.Forward(predicted, labels)
        
        # Calculate Mean Loss
        data_loss = np.mean(sample_losses)
        return data_loss


#==============================================================================
# Categorical Cross Entropy Loss
#==============================================================================
class CategoricalCrossEntropy_Loss(Loss):
    #--------------------------------------------------------------------------
    # Forward Propagation
    #--------------------------------------------------------------------------
    def Forward(self, y_pred, y_true):
        '''
        Parameters
        ----------
        y_pred : TYPE Numpy Array
            DESCRIPTION. Predicted values
        y_true : TYPE Numpy Array
            DESCRIPTION. Actual labels

        Returns
        -------
        sample_losses : TYPE Numpy Array
            DESCRIPTION. loss for each sample in the data 
        '''
        
        samples = len(y_pred)
        
        # Clip the data to avoid division by 0, clipping both sides to avoid mean dragging to any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
            
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
        negative_log_liklehoods = -np.log(correct_confidences)
        
        return negative_log_liklehoods
    
    #--------------------------------------------------------------------------
    # Backward Propagation
    #--------------------------------------------------------------------------
    def Backward(self, d_values, y_true):
        '''
        Parameters
        ----------
        d_values : TYPE Numpy Array
            DESCRIPTION. backprop data values
        y_true : TYPE Numpy Array
            DESCRIPTION. Actual labels

        Returns
        -------
        None.
        ''' 
        # Number of samples in the data
        samples = len(d_values)
        
        # Number of outputs in every sample
        labels = len(d_values[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            
        self.d_inputs = -y_true / d_values
        self.d_inputs = self.d_inputs / samples
        

#==============================================================================
# Softmax Activation and Categorical Cross Entropy Loss at once to Fasten Up 
#==============================================================================
class Activation_Softmax_Loss_CategoricalCrossEntropy():
    #--------------------------------------------------------------------------
    # Create Activation and Loss Function Object
    #--------------------------------------------------------------------------
    def __init__(self):
        self.activation = Softmax_Activation()
        self.loss = CategoricalCrossEntropy_Loss()
        
    #--------------------------------------------------------------------------
    # Forward Propagation
    #--------------------------------------------------------------------------
    def Forward(self, xTrain, yTrain, xVal, yVal):
        # Output Layers Activation Function
        self.activation.Forward(xTrain, xVal)
        
        # Set the Output
        self.output = self.activation.output
        self.val_output = self.activation.val_output
        
        # Calculate and return Loss values
        self.train_loss = self.loss.Calculate(self.output, yTrain)
        self.val_loss = self.loss.Calculate(self.val_output, yVal)
        
        return self.train_loss, self.val_loss
    
    #--------------------------------------------------------------------------
    # Backward Propagation
    #--------------------------------------------------------------------------
    def Backward(self, d_values, y_true):
        '''
        Parameters
        ----------
        d_values : TYPE Numpy Array
            DESCRIPTION. backprop data values
        y_true : TYPE Numpy Array
            DESCRIPTION. Actual labels

        Returns
        -------
        None.
        ''' 
        # Number of samples in the data
        samples = len(d_values)
        
        # If labels are not hot-encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # Copy values to safely modify them
        self.d_inputs = d_values.copy()
        
        # Calculate Gradients
        self.d_inputs[range(samples), y_true] -= 1
        
        # Normalize the Gradients
        self.d_inputs = self.d_inputs / samples
        
        
#==============================================================================
# Hinge Loss
#==============================================================================
class Hinge_Loss(Loss):
    #--------------------------------------------------------------------------
    # Forward Propagation
    #--------------------------------------------------------------------------
    def Forward(self, y_pred, y_true):
        '''
        Parameters
        ----------
        y_pred : TYPE Numpy Array
            DESCRIPTION. Predicted values
        y_true : TYPE Numpy Array
            DESCRIPTION. Actual labels

        Returns
        -------
        sample_losses : TYPE Numpy Array
            DESCRIPTION. loss for each sample in the data 
        '''
        y_true = y_true.reshape(-1, 1)
        margins = np.maximum(0, y_pred - y_pred[y_true] + 1)
        margins[y_true] = 0
        loss = np.sum(margins)
        
        return loss
    
        
#==============================================================================
# Common Accuracy Class 
#==============================================================================
class Accuracy:
    #--------------------------------------------------------------------------
    # Calculate Accuracy given predictions and labels
    #--------------------------------------------------------------------------
    def Calculate(self, y_pred, y_true):
        '''
        Parameters
        ----------
        y_pred : TYPE Numpy Array
            DESCRIPTION.Predicted Values
        y_true : TYPE Numpy Array
            DESCRIPTION. Actual Values

        Returns
        -------
        accuracy : TYPE Int
            DESCRIPTION. Scalar Value to determine accuracy of the model

        '''
        predictions = np.argmax(y_pred, axis=1)
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
        compare =  predictions == y_true
        accuracy = np.mean(compare)
        return accuracy
        

#==============================================================================
# Class Model
#==============================================================================
class Model():
    #--------------------------------------------------------------------------
    # Training the Model
    #--------------------------------------------------------------------------
    def Train(self, X, y, xVal, yVal, alpha=0.01, epochs=1000, print_every=1):    
        lamdas = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 1]
        minTrainCostArray = []             # Mimimum Train Costs from each lambda training
        minValCostArray = []               # Minimum Validation Costs from each lambda training
        
        for lamda in lamdas:
            print(f"Lamda: {lamda}")
            # Create dense layer with 11 input features and 20 output values
            self.dense_1 = Dense_Layer(12288, 3, lamda=lamda)
            
            # Create Sigmoid activation (to be used with upper dense layer), for classification
            self.activation_1 = Softmax_Activation()
            
            # Create Loss Functions
            self.loss_function = Activation_Softmax_Loss_CategoricalCrossEntropy()
            self.cost_function = CategoricalCrossEntropy_Loss()
            self.cost_function1 = Hinge_Loss()
            
            # Create Optimizer Function
            optimizer = GD_Optimizer(alpha=alpha)
            self.accuracy = Accuracy()
                
            trainCost = []               # Train Cost Array for cost from each iteration 
            valCost = []                 # Validation Cost Array for cost from each iteration 
            trainAccuracy = []           # Training Accuracy Array
            valAccuracy = []             # Validation Accuracy Array
            
            # Train in Loop
            for epoch in range(1, epochs+1):
                 
                # Perform forward pass on our training and validation data through this layer
                self.dense_1.Forward(X, xVal)
            
                # Perform forward pass through activation function, takes output of 2nd dense layer
                self.activation_1.Forward(self.dense_1.output, self.dense_1.val_output)
                
            
                # Calculate the training and validation data losses
                train_data_loss, val_data_loss = self.loss_function.Forward(self.dense_1.output, y, self.dense_1.val_output, yVal)
                
                # Calculate the training and validation data Hinge losses
                train_data_hinge_loss = self.cost_function1.Calculate(self.activation_1.output, y)
                val_data_hinge_loss = self.cost_function1.Calculate(self.activation_1.val_output, yVal)
                    
                # Calculate regularization penalty in all dense layers
                loss_regularization = self.loss_function.loss.Loss_Regularization(self.dense_1)
                
                # Calculate the total loss
                T_loss = train_data_loss + loss_regularization
                
                # Append all training and validation losses in the arrays initialzed above
                trainCost.append(train_data_loss)
                valCost.append(val_data_loss)
                
                # Calculate accuracy from output of activaion3 and actual labels
                train_accuracy = self.accuracy.Calculate(self.loss_function.output, y)
                val_accuracy = self.accuracy.Calculate(self.loss_function.val_output, yVal)
                
                trainAccuracy.append(train_accuracy)
                valAccuracy.append(val_accuracy)
                
                if not epoch % print_every:
                    print(f'epoch: {epoch}, ' +
                          f'Accuracy: {train_accuracy:.3f}, ' +  
                          f'Loss: {T_loss:.3f}, ' +
                          f'Regularization Loss: {loss_regularization:.3f}, ' +
                          f'Train Data Loss: {train_data_loss:.3f}, ' +
                          f'Validation Data Loss: {val_data_loss:.3f}, ' +
                          f'Train Hinge Loss: {train_data_hinge_loss:.3f}, ' +
                          f'Val Hinge Loss: {val_data_hinge_loss:.3f}, ' +
                          f'Learning Rate: {optimizer.alpha:.3f}') 
                    
                # Backward pass from right to left (chain rule)
                self.loss_function.Backward(self.loss_function.output, y)
                self.dense_1.Backward(self.loss_function.d_inputs)
                
                # Update weights and biases
                optimizer.Update_Params(self.dense_1)
                
            minTrainCost = np.min(trainCost)                               # Determining Minimum Training Cost Error
            minTrainCostArray.append(minTrainCost)                         # Append minimum train cost to the minimum train cost array 
            minVAlCost = np.min(valCost)                                   # Determining Minimum Validation Error
            minValCostArray.append(minVAlCost)                             # Append minimum validation cost to the minimum validation cost array
                
            print('Min Train Cost Error = ' + str(np.min(trainCost)))      # Determining Minimum Training Cost Error
            print('Min Validation Error = ' + str(np.min(valCost)))        # Determining Minimum Validation Error

            
        return lamdas, minTrainCostArray, minValCostArray

        
        
model = Model()
lamdas, minTrainCostArray, minValCostArray = model.Train(X_train, Y_train, X_val, Y_val, alpha=0.001, epochs=500, print_every=100)
   

plt.figure()
plt.plot(lamdas, minTrainCostArray)
plt.plot(lamdas, minValCostArray)
plt.xlabel('Lamda (λ)')
plt.ylabel('Cost Error')
plt.legend(["Training Cost", "Validation Error"], loc ="lower right")
plt.show()  