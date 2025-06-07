from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pymoo.core.problem import ElementwiseProblem
from sklearn.svm import SVR
import pyswarms as ps
from SwarmPackagePy import testFunctions as tf
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout, Flatten
from keras.layers import Flatten


main = tkinter.Tk()
main.title("Flyrock Distance Prediction using MOO + PSO + Hybrid ANN") #designing main screen
main.geometry("1300x1200")

global filename, X, Y, scaler, scaler1, pso, ann_model, dataset, svm_cls, dataset

#function to calculate accuracy and prediction sales graph
def calculateMetrics(algorithm, predict, test_labels):
    global scaler1
    mse_error = mean_squared_error(test_labels, predict)
    square_error = 1 - mse_error
    predict = predict.reshape(-1, 1)
    predict = scaler1.inverse_transform(predict)
    test_label = scaler1.inverse_transform(test_labels)
    predict = predict.ravel()
    test_label = test_label.ravel()    
    text.insert(END,algorithm+" MSE : "+str(mse_error)+"\n")
    text.insert(END,algorithm+" R2 : "+str(square_error)+"\n\n")
    print()
    for i in range(0, 10):
        text.insert(END,"True Flyrock Distance : "+str(test_label[i])+" Predicted Flyrock Distance : "+str(predict[i])+"\n")
    plt.figure(figsize=(5,3))
    plt.plot(test_label, color = 'red', label = 'True Flyrock Distance')
    plt.plot(predict, color = 'green', label = 'Predicted Flyrock Distance')
    plt.title(algorithm+' Flyrock Distance Prediction Graph')
    plt.xlabel('Test Data')
    plt.ylabel('Prediction')
    plt.legend()
    plt.show()    

def uploadDataset():
    text.delete('1.0', END)
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))

def processDataset():
    text.delete('1.0', END)
    global X, Y, dataset, scaler, scaler1
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler1 = MinMaxScaler(feature_range = (0, 1))
    dataset.fillna(0, inplace=True)#remove missing values
    dataset = dataset.values
    X = dataset[2:dataset.shape[0],1:dataset.shape[1]-1]
    X = scaler.fit_transform(X)
    Y = dataset[2:dataset.shape[0]:,dataset.shape[1]-1]
    text.insert(END,"Processed & Normalized Features = "+str(X))

#features optimization using MOO algorithm 
class MOO(ElementwiseProblem):
    def __init__(self, features, target):
        super().__init__(n_var=features.shape[1], n_obj=1)
        self.features = features
        self.target = target

    def _evaluate(self, x):
        # Calculate the objective function value for the given input x
        y = np.dot(x, self.features)
        return np.sum((y - self.target)**2)    
#PSO function
def f_per_particle(m, alpha):
    global X, Y, svm_cls
    total_features = X.shape[1]
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    svm_cls.fit(X_subset, Y) #pso function utilizing --SVM-- object to select optimized features and this features will be input to ANN to form hybrid algorithm
    P = (svm_cls.predict(X_subset) == Y).mean()
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j
def f(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)
def runPSO():
    text.delete('1.0', END)
    global X, Y, pso, svm_cls, scaler1
    svm_cls = SVR()#creating svm object
    text.insert(END,"Total features found in dataset before applying MOO Optimization & PSO Features Selection : "+str(X.shape[1])+"\n\n")
    moo = MOO(X, Y)#optimizing X features using MOO
    X = moo.features #getting optimized features
    options = {'c1': 0.9, 'c2': 0.9, 'w':0.5, 'k': 5, 'p':2}
    dimensions = X.shape[1] # dimensions should be the number of features
    optimizer = ps.discrete.BinaryPSO(n_particles=5, dimensions=dimensions, options=options) #CREATING PSO OBJECTS 
    cost, pso = optimizer.optimize(f, iters=10)#OPTIMIZING FEATURES
    X = X[:,pso==1] #select all features from X selected by pso
    Y = Y.reshape(-1,1)
    Y = scaler1.fit_transform(Y)
    text.insert(END,"Total features found in dataset after applying MOO Optimization & PSO Features Selection : "+str(X.shape[1])+"\n\n")

def runHybrdiANN():
    global X, Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    #now train & plot ANN Crop Yield prediction graph
    ann_model = Sequential()
    #adding ANN dense layer with 50 neurons to filter dataset 50 times
    ann_model.add(Dense(50, input_shape=(X_train.shape[1],)))
    ann_model.add(Activation('relu'))
    #dropout layer to remove irrrelevant features from dataset
    ann_model.add(Dropout(0.2))
    ann_model.add(Dense(50))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.2))
    ann_model.add(Dense(1))
    ann_model.compile(optimizer='adam', loss='mean_squared_error')
    ann_model.fit(X, Y, batch_size = 16, epochs = 500, validation_data=(X_test, y_test), verbose=1)
    predict = ann_model.predict(X_test)
    calculateMetrics("Hybrid MOO + PSO + SVM + ANN", predict, y_test)#call function to plot LSTM crop yield prediction
    

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Flyrock Distance Prediction using MOO + PSO + Hybrid ANN')
title.config(bg='honeydew2', fg='DodgerBlue2')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=27,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Flyrock Dataset", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=390,y=100)
processButton.config(font=font1) 

psoButton = Button(main, text="MOO Optimization & PSO + SVM Features Selection", command=runPSO)
psoButton.place(x=630,y=100)
psoButton.config(font=font1)

hybridButton = Button(main, text="Run Hybrid ANN Using PSO + SVM Features", command=runHybrdiANN)
hybridButton.place(x=10,y=150)
hybridButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=390,y=150)
exitButton.config(font=font1)

main.config(bg='LightSteelBlue3')
main.mainloop()
