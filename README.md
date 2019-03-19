# lane-detection-demo1
Environment： python3.*+Opencv3.*
---------------------------------

method :Use the traditional way to detect the lane,use the Opencv package<br>
(main function :GaussianBlur,Canny,HoughLinesP)

Beacause of the function HoughLinesP will return the image location :x1,y1,x2,y2 is the three-dimensional matrix 
So there are so many line's location will return 
I use the angel and length to find out the Valid array
#This is the simple way to find out this 
>videoget.py
>>selectrl.py


 A brief introduction to the code running process:
 ---------------------------------------------------
 ```python
 cap = cv2.VideoCapture('project_video.mp4')#get the video source file
 _,image = cap.read()
 mask = np.zeros_like(image)
 #the iamge is pictures per frame so you can use the Opencv to handle this video

 ```
 ```python
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
     #get the ROI area in the video 
     #the matrix is select by your video  
     #(this matrix is find out by yourself each video will have different matrix )
   ```
   


