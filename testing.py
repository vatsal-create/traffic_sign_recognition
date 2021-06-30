from utils import *
from preprocessing import load_data

def get_test_accuracy(print_test_data_shape=False,print_accuracy=False):
    [train_data,test_data,X_train,X_valid,Y_train,Y_valid]=load_data()
    image_list=[]
    label_list=[]
    c_path=r'C:\Users\Desktop\TRAFFIC SIGNAL\archive (1)'
    for i in test_data['Path']:
        p=c_path+'/'
        p=p+i
        img=cv2.imread(p,-1)
        img=cv2.resize(img,(32, 32),interpolation=cv2.INTER_NEAREST)
        image_list.append(img)
    image_list=np.array(image_list)

    for i in test_data['ClassId']:
        label_list.append(i)

    label_list=np.array(label_list)
    image_list=image_list/255
    if(print_test_data_shape):
        print(image_list.shape)

    model=get_model()
    res=model.predict_classes(image_list)
    if(print_accuracy):
        print(accuracy_score(res,label_list)*100)

def get_model(print_summary=False):
    model=Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation="softmax"))

    model.load_weights('Traffic_model.h5')
    model.trainable = False
    if(print_summary):
        print(model.summary())
    return model

def get_train_history():
    f = open('train_history.pckl', 'rb')
    train_history = pickle.load(f)
    f.close()
    return train_history

def display_model_history():
    train_history=get_train_history()
    plt.plot(train_history['accuracy'],label='Train_accuracy',color='g')
    plt.plot(train_history['val_accuracy'],label='Val_accuracy',color='b')
    plt.xlabel("EPOCHS")
    plt.ylabel("ACCURACY")
    plt.title("ACCURACY VS EPOCHS")
    plt.legend()
    plt.grid(False)
    plt.show()

    plt.plot(train_history['loss'], label='Train_loss', color='g')
    plt.plot(train_history['val_loss'], label='Val_loss', color='b')
    plt.xlabel("EPOCHS")
    plt.ylabel("LOSS")
    plt.title("LOSS VS EPOCHS")
    plt.legend()
    plt.grid(False)
    plt.show()

def manual_testing(image_path,show_image=False):
    i=[]
    img = cv2.imread(image_path,-1)
    I=img
    I=cv2.resize(I,(200,200),interpolation=cv2.INTER_AREA)
    if(show_image):
        cv2.imshow("Image",I)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_NEAREST)
    i.append(img)
    i=np.array(i)
    model=get_model()
    res=model.predict_classes(i)
    #print("Class Number:",res[0])
    #print("Signal Meaning:",dct[res[0]])
    return [res[0],dct[res[0]]]

def visualise_model(save_images=False):
    model=get_model()
    layered_img=visualkeras.layered_view(model)
    layered_img.show("LAYERED VIEW OF MODEL")
    graph_img=visualkeras.graph_view(model)
    graph_img.show("GRAPH VIEW OF MODEL")
    if(save_images):
        layered_img.save("layered_img.png")
        graph_img.save("gaph_img.png")

