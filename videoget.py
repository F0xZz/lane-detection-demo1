# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 06:52:29 2019

@author: huzejun
"""
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import numpy as np 
import cv2
import selectrl
def bypass_angle_filter(lines,low_thres,hi_thres):
    filtered_lines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            angle = abs(np.arctan((y2-y1)/(x2-x1))*180/np.pi)
            if angle > low_thres and angle < hi_thres:
                filtered_lines.append([[x1,y1,x2,y2]])
    return filtered_lines
def average_lines(img,lines,y_min,y_max):
    #return coordinates of the averaged lines
    hough_pts = {'m_left':[],'b_left':[],'norm_left':[],'m_right':[],'b_right':[],'norm_right':[]}
    if lines != None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                m,b = np.polyfit([x1,x2],(y1,y2),1)
                norm = ((x2-x1)**2+(y2-y1)**2)**0.5
                if m>0:#斜率right
                    hough_pts['m_right'].append(m)
                    hough_pts['b_right'].append(b)
                    hough_pts['norm_right'].append(norm)
                if m<0:
                    hough_pts['m_left'].append(m)
                    hough_pts['b_left'].append(b)
                    hough_pts['norm_left'].append(norm)
    if len(hough_pts['b_left'])!=0 or len(hough_pts['m_left'])!=0 or len(hough_pts['norm_left'])!=0:
        b_avg_left = np.mean(np.array(hough_pts['b_left']))
        m_avg_left = np.mean(np.array(hough_pts['m_left']))
        xmin_left = int((y_min-b_avg_left)/m_avg_left)
        xmax_left = int((y_max-b_avg_left)/m_avg_left)
        left_lane = [[xmin_left,y_min,xmax_left,y_max]]
    else:
        left_lane = [[0,0,0,0]]
    if len(hough_pts['b_right'])!=0 or len(hough_pts['m_right'])!=0 or len(hough_pts['norm_right'])!=0:
        b_avg_right = np.mean(np.array(hough_pts['b_right']))
        m_avg_right = np.mean(np.array(hough_pts['m_right']))
        xmin_right = int((y_min - b_avg_right)/m_avg_right)
        xmax_right = int((y_max-b_avg_right)/m_avg_right)
        right_lane = [[xmin_right,y_min,xmax_right,y_max]]
    else:
        right_lane = [[0,0,0,0]]
    return [left_lane,right_lane]

cap = cv2.VideoCapture('project_video.mp4')
rho=1
theta=np.pi/180
threshold=25
min_line_len=10
max_line_gap=10
low_thres=20
hi_thres=80
while(1):
    _,image = cap.read()
    mask = np.zeros_like(image)
#    imshape = image.shape
   
#get the ROI area in the video 
#the matrix is select by your video   
    if len(imshape)>2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,)*channel_count
        #print(ignore_mask_color)
    else:
        ignore_mask_color= 255 #fill the image in the black
    #choice the zone in the experience
    vertices1 = np.array([[(100,imshape[0]),(390,imshape[0]*0.65),\
                         (620,imshape[0]*0.65),(imshape[1],imshape[0]),\
                         (100,imshape[0])]],dtype=np.int32)
    vertices2 =np.array([[(550, 445), (700, 445), (1300, 720), (180, 720)]],dtype=np.int32)
    cv2.fillPoly (mask,vertices2,ignore_mask_color)
    masked_image =cv2.bitwise_and(image,mask)
       
    gray = cv2.cvtColor(masked_image,cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(gray_blur,10,180) 
   
    lines = cv2.HoughLinesP(edges,rho,theta,threshold,np.array([]),minLineLength=min_line_len,maxLineGap = max_line_gap)
#fill the intrest image zone in 1 use the and bitwise to show
#    cv2.imshow('mask',masked_image)
    hlines_img = np.zeros(imshape,dtype=np.uint8)
  
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(hlines_img,(x1,y1),(x2,y2),(0,255,0),3)

    h_lines = bypass_angle_filter(lines,low_thres,hi_thres)
#    print(h_lines.dtype)

    avg_hlines = average_lines(image,h_lines,int(image.shape[0]*0.7),image.shape[0])

    avg_img = np.zeros(imshape,dtype=np.uint8)
#    print(avg_hlines)
#    print(avg_hlines.dtype)
    rect = selectrl.newrect(avg_hlines)
    #rectnew = selectrl.selec(image,avg_hlines)
    rectnew = selectrl.selecchange(image,avg_hlines)
   # print(rect.shape)
   # print(rectnew.shape)
    y1=int(image.shape[0]*0.7)
    y2=image.shape[0]
    for lines in rect:
        for line in lines:
            for x1,x2,x3,x4 in line:
                mask = np.zeros_like(image)
                rect2=np.array([[(x1, y1), (x3, y1), (x4, y2), (x2, y2)]],dtype=np.int32)
                cv2.fillPoly (mask,rect2,(255,0,255))
                image = cv2.addWeighted(image,0.7,mask,0.3,0)
                cv2.line(image,(x1,y1),(x2,y2),(255,0,0),1)
                cv2.line(image,(x3,y1),(x4,y2),(255,0,0),1)
                #cv2.line(image,(x1,y1),(x3,y1),(0,255,0),3)
                #cv2.line(image,(x2,y2),(x4,y2),(0,0,255),3)


#    for line in avg_hlines:
#        for x1,y1,x2,y2 in line:
#            print('cc')
            #cv2.line(avg_img,(x1,y1),(x2,y2),(255,255,0),3)
#            print(line)
            #cv2.line(image,(x1,y1),(x2,y2),(255,255,0),3)
            
            #maskzeros = np.zeros_like(image)
            #ver = np.array([[(x2,y2),(x2,y2),(x1,image.shape[0]),(x2,image.shape[0])]],dtype = np.int32)
            #cv2.fillPoly(avg_img,ver,(0,255,0))
            #cv2.addWeighted(image,1,avg_img,0.3,0)
    #lineleft=selectrl.selec(image,avg_hlines)
    #for line in lineleft:
        #x1,x2,x3,x4=lineleft[]
        
        
#    cv2.imshow('edgs',edges)
    cv2.imshow('hline',avg_img)
#    cv2.imshow('a',avg_img)
    cv2.imshow('source',image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
    
        