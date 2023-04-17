import numpy as np
from pygame import mixer
import cv2
from tkinter import *
import tkinter.messagebox
from tensorflow.keras.models import load_model

root = Tk()
root.geometry('500x400')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH, expand=1)
root.title('Final Project')
frame.config(background='light blue')
label = Label(frame, text="Vehicle Driver's Drowsiness Detection", bg='light blue', font=('Times 20 bold'))
label.pack(side=TOP)
filename = PhotoImage(file="E:\Drowsiness Detection\demo1.png")  #path of the bg image of gui
background_label = Label(frame, image=filename)
background_label.pack(side=TOP)


def hel():
    help(cv2)


def Contri():
    tkinter.messagebox.showinfo("Contributors", "Manshi\nRaushan Raj")
def Aboutapp():
    tkinter.messagebox.showinfo("About The System","The Vehicle Driver Drowsiness Detection system is\na vehicle safety technology which helps prevent accidents\ncaused by the driver getting drowsy.")

menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="About", command=Aboutapp)

subm2 = Menu(menu)
menu.add_cascade(label="Contributors", command=Contri)



def exitt():
    exit()


def web():
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

def webrec():
    pass

def det():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    model = load_model(r'E:\Drowsiness Detection\data\train\models\model.h5')    #path of the model

    mixer.init()
    sound = mixer.Sound(r'E:\Drowsiness Detection\data\alarm.wav')        #path of the audio file
    cap = cv2.VideoCapture(0)
    Score = 0
    while True:
        ret, frame = cap.read()
        height, width = frame.shape[0:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=3)

        for (ex, ey, ew, eh) in eyes:
            # cv2.rectangle(frame,pt1=(ex,ey),pt2=(ex+ew,ey+eh), color= (255,0,0), thickness=3 )

            # preprocessing steps
            eye = frame[ey:ey + eh, ex:ex + ew]
            eye = cv2.resize(eye, (80, 80))
            eye = eye / 255
            eye = eye.reshape(80, 80, 3)
            eye = np.expand_dims(eye, axis=0)
            # preprocessing is done now model prediction
            prediction = model.predict(eye)

            # if eyes are closed
            if prediction[0][0] > 0.30:
                cv2.putText(frame, 'closed', (10, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                            color=(255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(frame, 'Score' + str(Score), (100, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1, color=(255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)
                Score = Score + 1
                if (Score > 15):
                    try:
                        sound.play()
                    except:
                        pass

            # if eyes are open
            elif prediction[0][1] > 0.90:
                cv2.putText(frame, 'open', (10, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                            color=(255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(frame, 'Score' + str(Score), (100, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1, color=(255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)
                Score = Score - 1
                if (Score < 0):
                    Score = 0

        cv2.imshow('frame', frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


but1 = Button(frame, padx=1, pady=1, width=25, bg='white', fg='blue', relief=RAISED , command=det,
              text='Open Camera & Detect', font=('Georgia 15 bold'))
but1.place(x=75, y=150)



but2 = Button(frame, padx=5, pady=5, width=5, bg='white', fg='blue', relief=RAISED , text='Exit', command=exitt,
              font=('Georgia 15 bold'))
but2.place(x=210, y=275)

root.mainloop()