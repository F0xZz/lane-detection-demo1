# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:30:59 2019

@author: huzejun
"""
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
def selec(image,lines):
    leftcorner=0
    lefttop=0
    righttop=0
    rightcorner=0
    x = image.shape[1]
    y = image.shape[0]
    xhalf = int(x/2)
#    print(xhalf)
    rect = []
    for line in lines:
        x1,y1,x2,y2=line[0]
        print(x1)
        print(x2)
        if(x1<xhalf&x2<xhalf):
            if(x1<x2):
                lefttop=x2
                leftcorner=x1
            else:
                lefttop=x1
                leftcorner=x2            
        elif(x1>xhalf&x2>xhalf):
            if(x1<x2):
                righttop=x1
                rightcorner = x2
            else:
                rightcorner=x1
                righttop =x2
          #rect.append([[lefttop,leftcorner,righttop,rightcorner]])   
        rect.append([[lefttop,leftcorner,righttop,rightcorner]])
    return rect
def selecchange(image,lines):
    x = image.shape[1]
    y = image.shape[0]
    xhalf = int(x/2)
#    print(xhalf)
    rect = []
    for line in lines:
        x1,y1,x2,y2=line[0]
          #rect.append([[lefttop,leftcorner,righttop,rightcorner]])   
        rect.append([x1,x2])
    return rect
def newrect(lines):
    locate = []
    lo = []
    retx = []
    for line in lines:
        for x1,y1,x2,y2 in line:
                #locate.append(x1,x2)
            locate.append(x1)
            locate.append(x2)
            
    lo.append([[locate]])
    
    return lo
            