import tkinter
from tkinter import filedialog
from tkinter import *
from utils import *
import PIL
from PIL import ImageTk,Image
from testing import manual_testing

window=tkinter.Tk()
window.geometry('1000x600')
img=cv2.imread("interface_background.jpg",-1)
img=cv2.resize(img,(1000,600),interpolation=cv2.INTER_NEAREST)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
c1=tkinter.Canvas(window,width=400,height=400)
c1.pack(fill="both",expand=True)
c1.create_image(0,0,image=img,anchor="nw")

def pre_image(path):
    [classId, meaning] = manual_testing(path)
    class_str = "ClassId:"
    class_str = class_str + str(classId)
    label=tkinter.Label(window, text=class_str, font=("Aerial Bold", 10), fg="black")
    t5=c1.create_window(450,300,anchor="nw",window=label)
    meaning = "Meaning:" + meaning
    label2=tkinter.Label(window, text=meaning, font=("Aerial  Bold", 10), fg="black")
    t6=c1.create_window(400,350,anchor="nw",window=label2)

def show_predict_button(path):
    button2=tkinter.Button(window,text="Predict",font=("Aerial Bold",15),fg="black",command=lambda : pre_image(path))
    t4=c1.create_window(875,290,anchor="nw",window=button2)

def upload_image():
    path=filedialog.askopenfilename()
    img2=cv2.imread(path,-1)
    img2=cv2.resize(img2,(250,250),interpolation=cv2.INTER_AREA)
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    img2=PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img2))
    label=tkinter.Label(window,image=img2)
    t1=c1.create_window(360,30, anchor="nw", window=label)

    show_predict_button(path)
    window.mainloop()

button1=tkinter.Button(window,text="Upload Test Image",font=("Aerial Bold",15),fg="black",command=upload_image)
t3=c1.create_window(375,540,anchor="nw",window=button1)
window.mainloop()
