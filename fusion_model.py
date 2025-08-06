from SimpleDL.Layer import Layer
from SimpleDL.Model import Model
import numpy as np

class fusion_model:
    def __init__(self):
        self.img_model=Model()
        self.num_model=Model()
        self.final_model=Model()
        self.batch_size=0
        self.epochs=0
        self.lr=0.01
        self.interval=10

    def fit_f(self,input_train1,input_train2,correct_train,batch_size,epochs,loss,lr,input_test1,input_test2,correct_test):
        self.batch_size=batch_size
        self.epochs=epochs
        self.lr=lr
        self.loss=loss

        self.final_model.loss=loss

        n_train=input_train1.shape[0]
        n_test=input_test1.shape[0]

        train_error_x=[]
        train_error_y=[]
        test_error_x=[]
        test_error_y=[]

        n_batch=n_train//self.batch_size

        sif=self.img_model.forwards
        snf=self.num_model.forwards
        sff=self.final_model.forwards
        sib=self.img_model.backwards
        snb=self.num_model.backwards
        sfb=self.final_model.backwards
        siu=self.img_model.update_wb
        snu=self.num_model.update_wb
        sfu=self.final_model.update_wb
        silon=self.img_model.layer_lists[-1].output_node
        index_random=np.arange(n_train)

        for i in range(self.epochs):
            print('Epoch:{}/{}'.format(i+1,self.epochs),end='')
            for j in range(n_batch):
                if j%100==0:
                    print('.',end='')
                mb_index=index_random[j*batch_size:(j+1)*batch_size]
                x1=input_train1[mb_index,:]
                x2=input_train2[mb_index,:]
                t=correct_train[mb_index,:]

                y1=sif(x1)
                y2=snf(x2)
                yf=sff(np.hstack([y1,y2]))

                g_xf=sfb(yf,t)
                g_x1=g_xf[:,:silon]
                g_x2=g_xf[:,silon:]
                sib(y1,g_x1)
                snb(y2,g_x2)

                siu()
                snu()
                sfu()
                
            ym=sif(input_train1)
            yn=snf(input_train2)
            y_f=sff(np.hstack([ym,yn]))
            error_train=self.final_model.get_error(y_f,correct_train,n_train)
            
            tym=sif(input_test1)
            tyn=snf(input_test2)
            ty_f=sff(np.hstack([tym,tyn]))
            error_test=self.final_model.get_error(ty_f,correct_test,n_test)

            test_error_x.append(i)
            test_error_y.append(error_test)
            train_error_x.append(i)
            train_error_y.append(error_train)

            print('Error_train:{:.5f} Error_test:{:.5f}'.format(error_train,error_test))
        return {'train_error_x':train_error_x,
                'train_error_y':train_error_y,
                'test_error_x':test_error_x,
                'test_error_y':test_error_y}
