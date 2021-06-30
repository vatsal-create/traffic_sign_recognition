from utils import *
from preprocessing import load_data,augment_data

[train_data,test_data,X_train,X_valid,Y_train,Y_valid]=load_data()
aug=augment_data()

model=Sequential()
model.add(Conv2D(32,kernel_size=(5,5),padding='same',activation='relu',input_shape=(32,32,3)))
model.add(Conv2D(32,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(rate=0.5))
model.add(Dense(43,activation="softmax"))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

epochs=20
fitted_model=model.fit(aug.flow(X_train,Y_train,batch_size=64),epochs=epochs,validation_data=(X_valid,Y_valid))

f = open('train_history.pckl', 'wb')
pickle.dump(fitted_model.history, f)
f.close()
model.save('Traffic_model.h5')
