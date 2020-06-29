# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:37:15 2020

@author: LENOVO
"""


import cv2, time
face_cascade=cv2.CascadeClassifier("C://Users//LENOVO//anaconda3//Lib//site-packages//cv2//data//haarcascade_frontalface_default.xml")

video=cv2.VideoCapture(0)
check, frame=video.read()
print(check)
print(frame)

time.sleep(3)
cv2.imshow('capturing',frame)

faces=face_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=5)
print(type(faces))
print(faces)

for x, y, w, h in faces:
    frame=cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0), 3)
    
#resized=cv2.resize(img, (int(img.shape[1]/7), (int(img.shape[0]/7) )))
cv2.imshow("Gray", frame)
a=1
while True:
    a=a+1
    check, frame=video.read()
    print(frame)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('capturing',gray)
    key=cv2.waitKey(1)
    
    if key==ord('q'):
        break

print(a)
video.release()
cv2.destroyAllWindows