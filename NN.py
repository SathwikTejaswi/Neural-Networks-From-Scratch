import numpy as np
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import rectified_linear_unit
from utils import grad_rectified_linear_unit
from utils import one_hot
from utils import categorical_cross_entropy
from utils import stable_softmax
from utils import decay_alpha
from utils import accuracy

class neural_network:
        
    def __init__(self,lr=0.001,loss_func='categorical_cross_entropy',MNIST=True,xtr=None,xts=None,ytr=None,yts=None):
        
        self.lr = lr
        self.loss_func = 'categorical_cross_entropy'
        
        if MNIST != True :
            assert(None not in [xtr,xts,ytr,yts])
            self.x_train = xtr
            self.x_test = xts
            self.y_train = ytr
            self.y_test = yts

        else :
            self.prepare_data()
            
    def prepare_data(self):
        
        self.get_data()
        self.x_train = (self.data_dict['x_train']).reshape(60000,784)
        self.x_test = (self.data_dict['x_test']).reshape(10000,784)
        self.y_train = (self.data_dict['y_train'])
        self.y_test = (self.data_dict['y_test'])
            
    def get_data(self,path='/Users/apple/Desktop/MNISTdata.hdf5'):
        
        print('')
        print('Fetching MNSIT Data')
        print('-------------------')
        data = h5py.File(path,'r')
        self.data_dict = {}
        
        for i in data:
            self.data_dict[i] = np.array(data[i])
        
        def get_data_stats(data_dict):
            
            print('The data has',list(data_dict.keys()))
            for i in data_dict:
                print(i, 'has shape :')
                print(np.array(data_dict[i]).shape)
                
        get_data_stats(self.data_dict)
        
            
    def print_model(self):
        print('')
        print('Model configurations are as follows :')
        print('-------------------------------------')
        
        print('layer 1')
        print('W1 has dim', self.W1.shape)
        print('b1 has dim', self.b1.shape)
        print('W2 has dim', self.W2.shape)
        print('b2 has dim', self.b2.shape)
        print('Input has dim', self.l1)
        print('Hidden layer has dim', self.l2)
        print('Output has dim', self.l3)
            
    def create_model(self, L2):
        
        self.l1 = 784
        self.l2 = L2
        self.l3 = 10

        self.W1 = np.random.randn(self.l2,self.l1)/np.sqrt(self.l1)
        self.b1 = np.zeros(shape = (self.l2,1))
        self.W2 = np.random.randn(self.l3,self.l2)/np.sqrt(self.l2)
        self.b2 = np.zeros(shape = (self.l3,1))
    
    def forward(self,x):
        
        self.z1 = np.array([self.W1.dot(x)]).transpose() + self.b1
        vec_rectified_linear_unit = np.vectorize(rectified_linear_unit)
        self.h = vec_rectified_linear_unit(self.z1)
        self.h = self.h.transpose()[0]
        self.z2 = np.array([self.W2.dot(self.h)]).transpose() + self.b2
        self.y_hat = stable_softmax(self.z2)
    
    def predict(self,x):
        y_hat_lab = np.zeros(shape = (len(x)))
        for i in range(len(x)):
            self.forward(x[i])
            y_hat_lab[i] = np.argmax(self.y_hat)
        return y_hat_lab
    
    def back_prop(self,Ytr1,Xtr1):
        
        y_hat = self.y_hat
        h = self.h
        z1 = self.z1
        
        true = one_hot(Ytr1)

        diff_outer = -(true - y_hat)

        del_b2 = diff_outer
        del_W2 = np.matmul(diff_outer,np.reshape(h,(1,self.l2)))
        DEL = self.W2.transpose().dot(diff_outer)

        NAB = np.multiply(DEL, grad_rectified_linear_unit(z1))
        del_b1 = NAB
        del_W1 = np.matmul(np.reshape(NAB, (self.l2,1)), np.reshape(Xtr1, (1,self.l1)))

        self.W2 = self.W2 - alpha * del_W2
        self.b2 = self.b2 - alpha * del_b2
        self.b1 = self.b1 - alpha * del_b1
        self.W1 = self.W1 - alpha * del_W1
        
    def save_weights(self):
        pd.DataFrame(self.W1).to_csv('W1.csv',index=False)
        pd.DataFrame(self.W2).to_csv('W2.csv',index=False)
        pd.DataFrame(self.b1).to_csv('b1.csv',index=False)
        pd.DataFrame(self.b2).to_csv('b2.csv',index=False)
        
    def load_model(self):
        self.W1 = np.array(pd.read_csv('W1.csv'))
        self.W2 = np.array(pd.read_csv('W2.csv'))
        self.b1 = np.array(pd.read_csv('b1.csv'))
        self.b2 = np.array(pd.read_csv('b2.csv'))
        
    def test_and_summarize(self):
        temp = self.x_test@self.W1.T+self.b1.T
        temp = np.clip(temp,a_min=0,a_max=temp.max())
        temp = temp@self.W2.T+self.b2.T
        temp = np.exp(temp)
        temp2 = temp.sum(axis=1)
        temp = temp/temp2.reshape(-1,1)
        preds = np.argmax(temp,axis=1)
        print('the testing accuracy of the classifier is :',sum(preds.reshape(-1,1) == self.y_test.reshape(-1,1))/len(self.y_test))
        
        
        
NN = neural_network()
NN.create_model(100)
NN.print_model()



x_learn, x_val, y_learning, y_val = train_test_split(NN.x_train, NN.y_train)
L = [i for i in range(0,len(x_learn))]
epochs = 14
print('')
print('------------------------')
print("start training the net")

for j in range(epochs):
    
    #IMPLEMETING SGD ALGORITHM
    
    alpha = decay_alpha(j)
    loss = 0
    np.random.shuffle(L)
    
    for i in L:
        NN.forward(x_learn[i])
        NN.back_prop(y_learning[i],x_learn[i])
        
        loss = loss + categorical_cross_entropy(NN.y_hat,y_learning[i])
    
    loss = loss/len(L)
    predicted_labels_validation = NN.predict(x_val)
    print('Epoch Summary for epoch:',j)
    print('Loss ->',loss[0])
    print('accuracy ->',accuracy(predicted_labels_validation,y_val))
    print('')
    
    

    
NN.test_and_summarize()