# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:53:03 2020

@author: LENOVO
"""

import cv2 
face_cascade=cv2.CascadeClassifier("C://Users//LENOVO//anaconda3//Lib//site-packages//cv2//data//haarcascade_frontalface_default.xml")

img=cv2.imread("C:\\Users\\LENOVO\\Desktop\\Face detection ML\\img3.png");

gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
print(type(faces))
print(faces)

for x, y, w, h in faces:
    img=cv2.rectangle(img,(x,y), (x+w, y+h), (0,255,0), 3)
    
#resized=cv2.resize(img, (int(img.shape[1]/7), (int(img.shape[0]/7) )))
cv2.imshow("Gray", img)
cv2.waitKey(0)
cv2.destroyAllWindows()    