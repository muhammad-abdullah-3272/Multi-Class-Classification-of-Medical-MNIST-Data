#------------------------------------------------------------------------------
# Importing Necessary libraires  
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
# Importing Datasets
#------------------------------------------------------------------------------
from Classes import Dense_Layer, Softmax_Activation, GD_Optimizer, CategoricalCrossEntropy_Loss, Hinge_Loss, Activation_Softmax_Loss_CategoricalCrossEntropy, Accuracy


 
#==============================================================================
# Class Model
#==============================================================================
class Model():
    #--------------------------------------------------------------------------
    # Training the Model
    #--------------------------------------------------------------------------
    def Train(self, X, y, xVal, yVal, lamda=0.03, alpha=0.01, epochs=100, print_every=1):    
        
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
        
            # Perform forward pass through activation function, takes output of 1st dense layer
            self.activation_1.Forward(self.dense_1.output, self.dense_1.val_output)
            
        
            # Calculate the training and validation data losses
            train_data_loss, val_data_loss = self.loss_function.Forward(self.dense_1.output, y, self.dense_1.val_output, yVal)
            
            # Calculate the training and validation data Hinge losses
            train_data_hinge_loss = self.cost_function1.Calculate(self.activation_1.output, y)
            val_data_hinge_loss = self.cost_function1.Calculate(self.activation_1.val_output, yVal)
                
            # Calculate regularization penalty in the layers
            loss_regularization = self.loss_function.loss.Loss_Regularization(self.dense_1)
            
            # Calculate the total loss
            T_loss = train_data_loss + loss_regularization
            
            # Append all training and validation losses in the arrays initialzed above
            trainCost.append(train_data_loss)
            valCost.append(val_data_loss)
            
            # Calculate accuracy from output of activaion3 and actual labels
            train_accuracy = self.accuracy.Calculate(self.loss_function.output, y)
            val_accuracy = self.accuracy.Calculate(self.loss_function.val_output, yVal)
            
            # Append all accuracies calculated
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
            
        return trainCost, valCost, trainAccuracy, valAccuracy

    #--------------------------------------------------------------------------
    # Testing the Trained Model
    #--------------------------------------------------------------------------
    def Test(self, xTest, yTest, xVal, yVal):
        self.dense_1.Forward(xTest, xVal)
        test_loss, val_loss = self.loss_function.Forward(self.dense_1.output, yTest, self.dense_1.val_output, yVal)
        test_accuracy = self.accuracy.Calculate(self.loss_function.output, yTest)
        val_accuracy = self.accuracy.Calculate(self.loss_function.val_output, yVal)
        print(self.loss_function.output[0:4])
        
        # Plotting The Data After Performing Test on Test Data
        classes = ["AbdomenCT", "Hand", "HeadCT"]
        plt.figure(figsize = (15,15))
        xTest = xTest.reshape(-1, 64, 64, 3)
        
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(xTest[i])
            plt.title(classes[yTest[i]], fontsize = 25)
        return test_accuracy, val_accuracy, test_loss, val_loss
