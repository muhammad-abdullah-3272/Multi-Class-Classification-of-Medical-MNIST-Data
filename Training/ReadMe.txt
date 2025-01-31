>> In order to train the model, run the Training.py file.
>> The following line of code in training.py is used to load the features and labels from the medical MNIST 
   dataset taken form kaggle.


>> If the dataset split needs to be changed then the following line of code in the Data_Preprocessing.py file 
   should be changed using:
	
        train, val, test = data[:22,:], data[22:26,:], data[26:,:]

	The above numbers represent the indices to split the data. Change these numbers to split the data. 
  
>> Finally, when the data is split into different sets the model is called and activation function is added, 
	and lambda value can be changed below as. The model is then set to particular functions and then finalized. 
	The user can change the above-mentioned parameters as;

	self.dense_1 = Dense_Layer(12288, 3, lamda=lamda)	
	self.activation_1 = Softmax_Activation()

>> Model is trained using below code. values of alpha and lambda, number of iterations to train the model and 
   print summary after every desired number can be set using.
	
	trainCost, valCost, trainAccuracy, valAccuracy = model.Train(X_train, Y_train, X_val, Y_val, lamda=0.03, alpha=0.001, epochs=300, print_every=100)

>> The model is then tested on test set and validation set. 