from utils import *

def load_data():
    train_data_path=r'C:\Users\Desktop\TRAFFIC SIGNAL\archive (1)\Train.csv'
    test_data_path=r'C:\Users\Desktop\TRAFFIC SIGNAL\archive (1)\Test.csv'
    train_data=pd.read_csv(train_data_path,usecols=['Width','Height','ClassId','Path'])
    test_data = pd.read_csv(test_data_path, usecols=['Width', 'Height', 'ClassId', 'Path'])

    height = 32
    widht = 32
    channels = 3

    image_data = []
    image_classId = []
    num_classes = 43
    train_image_path = r'C:\Users\Desktop\TRAFFIC SIGNAL\archive (1)\Train'

    for i in range(num_classes):
        curr_path = train_image_path + '/' + str(i)
        list = os.listdir(curr_path)
        for j in list:
            path = curr_path + '/' + j
            img = cv2.imread(path, -1)
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_NEAREST)
            image_data.append(img)
            image_classId.append(i)
    image_data = np.array(image_data)
    image_classId = np.array(image_classId)

    image_classId = keras.utils.to_categorical(image_classId, 43)
    X_train, X_valid, Y_train, Y_valid = sklearn.model_selection.train_test_split(image_data, image_classId,test_size=0.2, random_state=42,shuffle=True)
    X_train = X_train / 255
    X_valid = X_valid / 255

    return [train_data,test_data,X_train,X_valid,Y_train,Y_valid]

def analyise_data():
    [train_data,test_data,_,_,_,_]=load_data()
    show_data=train_data.groupby('ClassId').count()
    plt.bar(show_data.index,show_data['Width'],width=0.5,color='b')
    plt.title("COUNT OF EACH CLASS ID")
    plt.xlabel("Class Id")
    plt.ylabel("Count")
    plt.show()

    show_data=train_data.groupby('Width').count()
    plt.bar(show_data.index,show_data['Height'],width=0.5)
    plt.title("COUNT OF EACH WIDTH VALUE")
    plt.xlabel("WIDTH")
    plt.ylabel("COUNT")
    plt.show()

    show_data=train_data.groupby('Height').count()
    plt.bar(show_data.index,show_data['Width'],width=0.5)
    plt.title("COUNT OF EACH HEIGHT VALUE")
    plt.xlabel("HEIGHT")
    plt.ylabel("COUNT")
    plt.show()

def augment_data():
    aug = ImageDataGenerator(rotation_range=10, zoom_range=0.15, width_shift_range=0.1, height_shift_range=0.1,shear_range=0.15, horizontal_flip=False, vertical_flip=False, fill_mode="nearest")
    return aug

def print_data():
    [train_data, test_data, X_train, X_valid, Y_train, Y_valid]=load_data()
    print("Train Data")
    print(train_data.head())
    print("No of unique classes:", train_data['ClassId'].nunique())

    print("Training Data Shape X and Y:", X_train.shape, Y_train.shape)
    print("Validation Data Shape X and Y:", X_valid.shape, Y_valid.shape)
    print("Tain Label head")
    print(Y_train[0:5])
