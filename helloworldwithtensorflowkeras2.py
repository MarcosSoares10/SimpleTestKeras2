from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()

img_rows = x_train[0].shape[0]
img_columns = x_train[1].shape[0]

x_train = x_train.reshape(x_train.shape[0],img_rows,img_columns,1)
x_test = x_test.reshape(x_test.shape[0],img_rows,img_columns,1)

num_classes = 10
input_shape = (img_rows,img_columns,1)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

x_train = x_train / 255.0
x_test = x_test / 255.0 

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense

model = Sequential(
        [
        Input(shape = input_shape),
        Conv2D(32,kernel_size = (3,3),activation="relu"),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64,kernel_size=(3,3),activation="relu"),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dropout(0.5),
        Dense(num_classes,activation="softmax"),
        ]    
)

print(model.summary())

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"]
    )

history = model.fit(
    x_train,
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.1
)

score = model.evaluate(x_test,y_test,verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

hist_dict = history.history
loss_values = hist_dict["loss"]
val_loss_values = hist_dict["val_loss"]

epochs = range(1,len(loss_values)+1)

line1 = plt.plot(epochs,val_loss_values,label="Validation/Test loss")
line2 = plt.plot(epochs,loss_values,label = "Training loss")

plt.setp(line1,linewidth=2.0,marker="+",markersize=10.0)
plt.setp(line2,linewidth=2.0,marker="4",markersize=10.0)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

acc_values = hist_dict["accuracy"]
val_acc_values = hist_dict["val_accuracy"]

epochs = range(1,len(acc_values)+1)

line1 = plt.plot(epochs,val_acc_values,label="Validation/Test acc")
line2 = plt.plot(epochs,acc_values,label = "Training acc")

plt.setp(line1,linewidth=2.0,marker="+",markersize=10.0)
plt.setp(line2,linewidth=2.0,marker="4",markersize=10.0)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()

for i in range(0,10):
  random = np.random.randint(0,len(x_test))
  inputimg = x_test[random]
  inputimg = inputimg.reshape(1,28,28,1)

  result = str(
      model.predict_classes(inputimg,1,verbose=0)[0]
  )
  print(result)