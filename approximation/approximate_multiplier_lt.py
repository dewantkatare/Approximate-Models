# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 12:23:27 2023

@author: sshakibhamedan
"""

import numpy as np
multer1=np.load("../Approximate Multiplier/Approximate Mult1.npy")
multer2=np.load("../Approximate Multiplier/Approximate Mult2.npy")
multer3=np.load("../Approximate Multiplier/Approximate Mult3.npy")
multer4=np.load("../Approximate Multiplier/Approximate Mult4.npy")
multer5=np.load("../Approximate Multiplier/Approximate Mult5.npy")
multer6=np.load("../Approximate Multiplier/Approximate Mult6.npy")
multer7=np.load("../Approximate Multiplier/Approximate Mult7.npy")
multer8=np.load("../Approximate Multiplier/Approximate Mult8.npy")

def My_Arg_Mult(a,b,t=1):
  a=np.array(a)
  b=np.array(b)
  a_shape=np.shape(a)
  b=np.reshape(b,a_shape)
  res=np.zeros(a_shape)
  if t==1:
      multer=multer1
  if t==2:
      multer=multer2
  if t==3:
      multer=multer3
  if t==4:
      multer=multer4
  if t==5:
      multer=multer5
  if t==6:
      multer=multer6
  if t==7:
      multer=multer7
  if t==8:
      multer=multer8
  if len(a_shape)==1:
    for i in range(np.shape(a)[0]):
      res[i]=multer[int(a[i])+128,int(b[i])+128]
  if len(a_shape)==2:
    for i in range(a_shape[0]):
      for j in range(a_shape[1]):
        res[i,j]=multer[int(a[i,j])+128,int(b[i,j])+128]
  return res
#########################################################
def My_matmul(a,b,t=1):
  a=np.array(a)
  b=np.array(b)
  a_shape=np.shape(a)
  b_shape=np.shape(b)
  res=np.zeros([a_shape[0],b_shape[1]])
  for i in range(a_shape[0]):
    for j in range(b_shape[1]):
      res[i,j]=np.sum(My_Arg_Mult(a[i,:],b[:,j],t))
  return res
#########################################################
def Myconv2d(a,b,t=1):
  #import numpy as np
  a=np.array(a)
  b=np.array(b)
  a_shape=np.shape(a)
  b_shape=np.shape(b)
  res_shape1=np.abs(a_shape[0]-b_shape[0])+1
  res_shape2=np.abs(a_shape[1]-b_shape[1])+1
  res=np.zeros([res_shape1,res_shape2])
  for i in range(res_shape1):
    for j in range(res_shape2):
      res[i,j]=np.sum(My_Arg_Mult(np.flip(b),a[i:i+b_shape[0],j:j+b_shape[1]],t))
  return res
    
