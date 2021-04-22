
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import os
import threading

model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7,activation='softmax'))
model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)

emotion_dict={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
cur_path=os.path.dirname(os.path.abspath(__file__))
emoji_path={0:"C:/Users/Anil Sharma/PycharmProjects/untitled6/emotion_detection/emojis/angry.png",1:cur_path+"/emojis/disgust.png",2:cur_path+"/emojis/fear.png",
            3:cur_path+"/emojis/happy.png",4:cur_path+"/emojis/neutral.png",5:cur_path+"/emojis/sad.png",
            6:cur_path+"/emojis/surprise.png"}

global last_frame1
last_frame1=np.zeros((480,640,3),dtype=np.uint8)
global cap1
show_text=[0]
global frame_no

def show_subject():
    cap1=cv2.VideoCapture(0)
    print('hello')
    if not cap1.isOpened():
        print("Cannot open camera")
    global frame_no
    length=int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_no +=1
    print(length,frame_no)

    cap1.set(1,frame_no)
    ret,frame=cap1.read()
    frame=cv2.resize(frame,(600,500))
    face_haar_cascade = cv2.CascadeClassifier("c:/Users/Anil Sharma/Downloads/face_detection.xml")  # Load haar classifier
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32,
                                               minNeighbors=5)  # detectMultiScale returns rectangles
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(255,0,0),2)
        roi_gray=gray_img[y:y+h,x:x+w]
        cropped=np.expand_dims(np.expand_dims(cv2.resize(roi_gray,(48,48)),-1),0)
        prediction=model.predict(cropped)
        maxindex=int(np.argmax(prediction))
        cv2.putText(frame,emotion_dict[maxindex],(x+20,y-60),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,(255,255,255),
                    2,cv2.LINE_AA)
        show_text[0]=maxindex
    if ret is None:
        print("problem")
    elif ret:
        global last_frame1
        last_frame1=frame.copy()
        pic=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=Image.fromarray(pic)
        imgtk=ImageTk.PhotoImage(img)
        lmain.imgtk=imgtk
        lmain.configure(image=imgtk)
        root.update()

    if cv2.waitKey(1) & 0xFF==ord('q'):
        cap1.release()
        cv2.destroyAllWindows()
        exit()

def show_emoji():
    frame2 = cv2.imread(emoji_path[show_text[0]])
    pic2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(pic2)
    imgtk2 = ImageTk.PhotoImage(img2)
    lmain2.imgtk2 = imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',45,'bold'))
    lmain2.configure(image=imgtk2)
    root.update()
    lmain2.after(10,show_emoji)
if __name__== '__main__':
    frame_no=0
    root=tk.Tk()
    lmain=tk.Label(master=root,padx=50,bd=10)
    lmain2=tk.Label(master=root,bd=10)
    lmain3=tk.Label(master=root,bd=10,fg='#CDCDCD',bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50,y=250)
    lmain3.pack()
    lmain3.place(x=960,y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900,y=350)
    root.title("emoji")
    root.geometry('1400x900+100+10')
    root['bg']='black'
    exitButton=Button(root,text='Quit',fg='red',command=root.destroy,font=('arial',25,'bold')).pack(side=BOTTOM)
    threading.Thread(target=show_subject).start()
    threading.Thread(target=show_emoji).start()

    root.mainloop()
