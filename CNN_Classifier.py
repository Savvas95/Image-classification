#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import os


# In[2]:


input_dir = 'Data/clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        image = cv2.imread(img_path)
        if image is not None :
            image = cv2.resize(image, (111, 49))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # normalization of the pixel values
            image = image / 255.0
        
            data.append(image)
            labels.append(category_idx)

data = np.array(data)
labels = np.array(labels)


# In[3]:


image.shape


# In[4]:


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, shuffle=True, random_state=42)


# In[5]:


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation ='relu', input_shape = image.shape))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))


model.compile(optimizer = optimizers.Adam(learning_rate = 0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[6]:


model.summary()


# In[7]:


num_epochs = 10
batch_size = 32

history = model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch_size, validation_split = 0.25)


# In[8]:


# Evaluate the model on the test set
results = model.evaluate(x_test, y_test)

# Print the evaluation results (accuracy and loss)
print(f"Test Accuracy: {results[1] * 100:.2f}%")
print(f"Test Loss: {results[0]:.4f}")

# Make predictions on a few random examples
num_examples_to_predict = 10
sample_indices = np.random.choice(len(x_test), num_examples_to_predict, replace=False)
sample_images = x_test[sample_indices]
sample_labels = y_test[sample_indices]

# Predictions
predictions = model.predict(sample_images)

# Convert predictions to binary labels
binary_predictions = (predictions > 0.5).astype(int)

# Display the actual labels, predicted labels, and sample images
for i in range(num_examples_to_predict):
    actual_label = "not_empty" if sample_labels[i] == 1 else "empty"
    predicted_label = "not_empty" if binary_predictions[i] == 1 else "empty"
    
    print(f"Example {i + 1}:")
    print(f"Actual Label: {actual_label}, Predicted Label: {predicted_label}")
    
    # Display the image (assuming matplotlib is available)
    plt.imshow(sample_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
    plt.show()


# In[9]:


# Save the model and weights
model_name = 'Convolutional_Model.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Save model and weights at %s ' % model_path)

# Save the model architecture to disk
model_json = model.to_json()
with open(os.path.join(save_dir, "model_json.json"), "w") as json_file:
    json_file.write(model_json)

# Load the model from disk
json_file_path = os.path.join(save_dir, 'model_json.json')
json_file = open(json_file_path, 'r')
loaded_model_json = json_file.read()
json_file.close()


# # #

# # The End

# In[ ]:




