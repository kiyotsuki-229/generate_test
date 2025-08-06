import numpy as np

class Layer:
    def __init__(self,input_node,output_node,activation_function):
        self.input_node=input_node
        self.output_node=output_node
        self.wb_width=0.1
        self.w=self.wb_width*np.random.randn(input_node,output_node)
        self.b=self.wb_width*np.random.randn(output_node)
        self.activation_function=activation_function


    def forward(self,x):
        self.x=x
        try:
            self.nout=np.dot(x,self.w)+self.b
        except:
            print(x.shape,self.w.shape,self.b.shape)

        if self.activation_function=='ReLU':
            y=np.where(self.nout<=0,0,self.nout)
        elif self.activation_function=='sigmoid':
            y=np.exp(np.minimum(self.nout,0))/(1+np.exp(-np.abs(self.nout)))
        elif self.activation_function=='softmax':
            y=np.exp(self.nout)/np.sum(np.exp(self.nout),axis=1,keepdims=True)
        elif self.activation_function=='identity':
            y=self.nout
        return y

    def backward(self,y,grad_y):
        if self.activation_function=='ReLU':
            delta=grad_y*np.where(self.nout<=0,0,1)
        elif self.activation_function=='sigmoid':
            sy=np.exp(np.minimum(self.nout,0))/(1+np.exp(-np.abs(self.nout)))
            delta=grad_y*sy*(1-sy)
        elif self.activation_function=='softmax':
            delta=y-grad_y
        elif self.activation_function=='identity':
            delta=y-grad_y
        self.grad_w=np.dot(self.x.T,delta)
        self.grad_b=np.sum(delta,axis=0)
        self.grad_x=np.dot(delta,self.w.T)
        return self.grad_x

    def update(self,eta):
        self.w-=eta*self.grad_w
        self.b-=eta*self.grad_b
