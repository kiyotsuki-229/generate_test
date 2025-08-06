from SimpleDL.Model import Model
from SimpleDL.Util import load_model
import numpy as np
import cv2
from PIL import Image

N='2'

num=input('number')
num2=np.zeros(10)
num2[int(num)]=1

for i in range(5):
    mi=load_model('mnist_generating{}_{}img.model'.format(N,str(i)))
    mn=load_model('mnist_generating{}_{}num.model'.format(N,str(i)))
    mf=load_model('mnist_generating{}_{}fin.model'.format(N,str(i)))

    if i==0:
        randimg=np.random.randint(0,256,784)
    else:
        randimg=gene3
    gene1=mi.predict(randimg)
    gene2=mn.predict(num2)
    gene3=mf.predict(np.hstack([gene1,gene2]))

img=gene3.reshape([28,28])
img*=255.0
cv2.imwrite('generated1_5.png',img)
