## VE593 Project3

### Project description

In this project we are required to model ANN and CNN to learn the pattern of data from Fashion-MNIST and German Traffic signs. 

## System requirement

Program and run in Jupyternote book with python 3.8 and

```python
import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import datetime
import cv2
import numpy as np
import os
from PIL import Image
```

## Part1. Fashion-MNIST

The file is named as **fashion-mnist.ipynb** there are several functions within it.

1. `dtsvg(index,x_train,y_train)`
  
  this function is used to generate one picture for each category.
  
  ```
  input: index, dataset
  output: pictures saved
  ```
  
2. The model is writen in the script cells, you can simply run the cell. 
  

## Part2. Traffic Sign

The file is named as **traffic-sign.ipynb** there are different methods within it

1. `imgLd(dn)`
  
  ```
  input: the folder name of the set("train", "test", "valid")
  output: a list of data
  ```
  
2. `labLd(dn)`
  
  ```
  input: the folder name of the set("train", "test", "valid")
  output: a list of label
  ```
  
3. The data exploration part is writen in script cells, you can simply run the cell. Moreover, in this part, I have
```python
print(x_test[1].shape)
for i in range(1,len(x_test)):
    if x_test[i].shape!=x_test[i-1].shape:
        print("Wrong")
print("Yes, all shapes are the same")
```
to check the shapes for the input pictures. 
  
4. `dtsvg(index,x_train,y_train)`
  
  ```
  input: index of the category, dataset
  output: pictures saved
  ```
  
5. `imgPr(name)`
  
  It will resize the data and do prediction
  
  ```
  input: the name of the test picture file
  output: the shape of the picture input and the prediction matrix as well as the final prediction result. 
  ```
  
6. `gimgLd(dn)`
  
  It will resize the data and convert the color of the data into grey, and save the image as a list
  
  ```
  input: the name of the test picture file
  output: a list of data
  ```
  
7. `gimgPr(name)`
  
   It will resize the data and convert the color of the data into grey and do prediction
  
  ```
  input: the name of the test picture file
  output: the shape of the picture input and the prediction matrix as well as the final prediction result. 
  ```

8. All the models are written in the script cell, you can simply run the cells to train, evaluate, print the model. 
