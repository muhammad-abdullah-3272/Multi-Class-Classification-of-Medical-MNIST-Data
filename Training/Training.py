#------------------------------------------------------------------------------
# Importing Necessary libraires  
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
# Importing Datasets
#------------------------------------------------------------------------------
from Data_Preprocessing import Data

#------------------------------------------------------------------------------
# Importing Model
#------------------------------------------------------------------------------
from Model import Model


# Instantiate the class Data
data = Data()

# Load the Medical MNIST Data
X_train, Y_train, X_val, Y_val, X_test, Y_test = data.Load_Data()

# Scale the Data
X_train, X_val, X_test = data.Scale(X_train, X_val, X_test)

# == Plot a sample Image (optional) == #
# data.Plot_Sample(X_train, Y_train, 0)

# Instantiate the class Model
model = Model()

# Train the Model
trainCost, valCost, trainAccuracy, valAccuracy = model.Train(X_train, Y_train, X_val, Y_val, lamda=0.03, alpha=0.001, epochs=300, print_every=100)
   

plt.figure()
plt.plot(trainCost)
plt.plot(valCost)
plt.title("Training and Validation Costs")
plt.xlabel('Iterations')
plt.ylabel('Cost Error')
plt.legend(["Training Cost", "Validation Error"], loc ="upper right")
plt.show()

plt.figure()
plt.plot(trainAccuracy)
plt.plot(valAccuracy)
plt.title("Training and Validation Accuracies")
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend(["Training", "Validation"], loc ="upper right")
plt.show()


# Test the Optimized Trained Model
test_accuracy, val_accuracy, test_loss, val_loss = model.Test(X_test, Y_test, X_val, Y_val)
print(f"Test Accuracy {test_accuracy:.3f}, Loss {test_loss:.3f}")
print(f"Val Accuracy {val_accuracy:.3f}, Loss {val_loss:.3f}")