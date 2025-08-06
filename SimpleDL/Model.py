from SimpleDL.Layer import Layer
import numpy as np

class Model:
    def __init__(self):
        self.layer_lists=[]
        self.batch_size=0
        self.epochs=0
        self.lr=0.01
        self.interval=10

    def add(self,input_node,output_node,activation_function):
        l=Layer(input_node,output_node,activation_function)
        self.layer_lists.append(l)

    def summary(self):
        for l in self.layer_lists:
            print('I:{},O:{},A:{}'.format(l.input_node,
                                         l.output_node,
                                         l.activation_function))

    def forwards(self,x):
        for l in self.layer_lists:
            x=l.forward(x)
        return x

    def backwards(self,y,x):
        for l in self.layer_lists[::-1]:
            x=l.backward(y,x)
        return x

    def get_error(self,y,t,data_count):
        if self.loss=='cross_entropy_error':
            return -np.sum(t*np.log(y+1e-7))/data_count
        elif self.loss=='squared_error':
            return 1.0/2.0*np.sum(np.square(y-t))/data_count

    def update_wb(self):
        for l in self.layer_lists:
            l.update(self.lr)

    def fit(self,input_train,correct_train,batch_size,epochs,loss,lr,input_test,correct_test):
        self.batch_size=batch_size
        self.epochs=epochs
        self.lr=lr
        self.loss=loss

        n_train=input_train.shape[0]
        n_test=input_test.shape[0]

        train_error_x=[]
        train_error_y=[]
        test_error_x=[]
        test_error_y=[]

        n_batch=n_train//self.batch_size

        nparange=np.arange
        sfw=self.forwards
        sbw=self.backwards
        sge=self.get_error

        index_random=nparange(n_train)

        for i in range(self.epochs):
            print('Epoch:{}/{}'.format(i+1,self.epochs,),end='')
            for j in range(n_batch):
                if j%100==0:
                    print('.',end='')
                mb_index=index_random[j*batch_size:(j+1)*batch_size]
                x=input_train[mb_index,:]
                t=correct_train[mb_index,:]

                y=sfw(x)
                sbw(y,t)

                self.update_wb()

            y=sfw(input_train)
            error_train=sge(y,correct_train,n_train)
            y=sfw(input_test)
            error_test=sge(y,correct_test,n_test)

            test_error_x.append(i)
            test_error_y.append(error_test)
            train_error_x.append(i)
            train_error_y.append(error_train)

            print('Error_train:{:.5f} Error_test:{:.5f}'.format(error_train,error_test))
        return {'train_error_x':train_error_x,
                'train_error_y':train_error_y,
                'test_error_x':test_error_x,
                'test_error_y':test_error_y}
    def predict(self,x):
        y=self.forwards(x)
        return y
