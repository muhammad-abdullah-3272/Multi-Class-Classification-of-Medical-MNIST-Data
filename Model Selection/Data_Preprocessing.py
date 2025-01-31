#------------------------------------------------------------------------------
# Importing Necessary libraires  
#------------------------------------------------------------------------------
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(7)

#==============================================================================
# Initializing Class for Data Preprocessing
#==============================================================================
class Data:
    #--------------------------------------------------------------------------
    # Load Dataset 
    #--------------------------------------------------------------------------
    def Load_Data(self):
    
        # Collecting the path of the Medica MNIST data
        for dirname, _, filenames in os.walk('/kaggle/input'):
            for filename in filenames:
                print(os.path.join(dirname, filename))
                
        # Classes in the dataset      
        classes = {'AbdomenCT': 0, 'Hand': 1, 'HeadCT': 2}
        
        # Image size
        IMAGE_WIDTH=64
        IMAGE_HEIGHT=64
        
        # Directories of the 
        abdomen_DIR='kaggle/input/medical-mnist/AbdomenCT'
        hand_DIR='kaggle/input/medical-mnist/Hand'
        headct_DIR = 'kaggle/input/medical-mnist/HeadCT'
        
        #----------------------------------------------------------------------
        # Assigning the Labels to Classes
        #----------------------------------------------------------------------        
        def Assign_Label(img, class_type):
            return class_type
        
        X = []
        y = []
        
        #----------------------------------------------------------------------
        # Stacking All classes to make the dataset
        #---------------------------------------------------------------------- 
        def Make_Data(class_type,DIR):
            
            # Fetching Features and Labels from the Directories from Medical MNIST folder
            for img in tqdm(os.listdir(DIR)):
                # Assigning the label to the classes
                label = Assign_Label(img,class_type)
                
                # Fetching path for the images from directories
                path = os.path.join(DIR,img)
                
                # images are fetched from path and resized to a specific resolution and 
                # color channels are also determined. Images are fetched one at a time
                img = cv2.imread(path,cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMAGE_WIDTH,IMAGE_HEIGHT))
                
                # Append Features and Labels to the arrays
                X.append(np.array(img))
                y.append(str(label))
                
        # Make data and get classes from directories
        Make_Data(classes.get('AbdomenCT'), abdomen_DIR)
        Make_Data(classes.get('Hand'), hand_DIR)
        Make_Data(classes.get('HeadCT'), headct_DIR)
        
        # Make the Features and Labels numpy arrays
        y = np.array(y)
        X = np.array(X)
        
        # Reshape Features and Labels
        X = X.reshape(-1, 12288)
        y = y.reshape(-1, 1)
        print("Features: ", X.shape, ", Labels: ", y.shape)
        
        # Concatenate Features and Labels to make a dataset for splittng
        data = np.concatenate((X,y),axis=1)
        print("Dataset: ", data.shape)
        
        # Randomly shuffle the Data and Split it into Train, Validation and Test Data with 70,15 and 15 %
        np.random.shuffle(data)
        train, val, test = data[:22,:], data[22:26,:], data[26:,:]
        
        print("Train Data: ", train.shape, ", Val Data: ", val.shape, ", Test Data: ", test.shape)
        
        # Extract Features and Labels from Training Data 
        X_train = train[:,:12288]
        Y_train = train[:,12288:12289]
        Y_train = Y_train.reshape(-1,)
        Y_train = Y_train.astype(int)
        print("X Train: ", X_train.shape, ", Y Train: ", Y_train.shape)
        
        # Extract Features and Labels from Training Data
        X_val = val[:,:12288]
        Y_val = val[:,12288:12289]
        Y_val = Y_val.reshape(-1,)
        Y_val = Y_val.astype(int)
        print("X Val: ", X_val.shape, ", Y Val: ", Y_val.shape)
        
        # Extract Features and Labels from Training Data
        X_test = test[:,:12288]
        Y_test = test[:,12288:12289]
        Y_test = Y_test.reshape(-1,)
        Y_test = Y_test.astype(int)
        print("X Test: ", X_test.shape, ", Y Test: ", Y_test.shape)
        
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
        
    #--------------------------------------------------------------------------
    # Plot a Sample Image from the Data
    #--------------------------------------------------------------------------
    def Plot_Sample(self, X, Y, index):
        classes = ["AbdomenCT", "Hand", "HeadCT"]
        plt.figure(figsize = (15,2))
        X = X.reshape(-1, 64, 64, 3)
        plt.imshow(X[index])
        plt.xlabel(classes[Y[index]])
            
    #--------------------------------------------------------------------------
    # Data Scaling 
    #--------------------------------------------------------------------------
    def Scale(self,X_train, X_val, X_test):            
        X_train = X_train.astype('float64')/255
        X_val = X_val.astype('float64')/255
        X_test = X_test.astype('float64')/255
            
        return X_train, X_val, X_test

