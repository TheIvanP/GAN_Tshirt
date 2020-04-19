#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#%%
fmnmist = tf.keras.datasets.fashion_mnist

(X_train, y_train), (X_test,y_test) = fmnmist.load_data()
#(X_train, y_train), (X_test,y_test) = tf.keras.datasets.mnist.load_data()
#%%
plt.imshow(X_train[0])
# %%
y_train==0
# %%
only_zeros = X_train[y_train==0]
# %%
only_zeros.shape
# %%
from tensorflow import keras
discriminator = keras.models.Sequential()
discriminator.add(keras.layers.Flatten(input_shape = [28,28]))
discriminator.add(keras.layers.Dense(150,activation='relu'))
discriminator.add(keras.layers.Dense(100,activation='relu'))

discriminator.add(keras.layers.Dense(1,activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

# %%
codings_size = 100
generator = keras.models.Sequential()
generator.add(keras.layers.Flatten(input_shape = [codings_size]))
generator.add(keras.layers.Dense(100,activation='relu'))
generator.add(keras.layers.Dense(150,activation='relu'))
generator.add(keras.layers.Dense(784,activation='relu'))
generator.add(keras.layers.Reshape([28,28]))
# %%
GAN = keras.models.Sequential([generator,discriminator])
discriminator.trainable = False
GAN.compile(loss = 'binary_crossentropy',optimizer='adam')
#%%
batch_size = 32
my_data = only_zeros
dataset = tf.data.Dataset.from_tensor_slices(my_data).shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
epochs = 100
GAN.layers[0].summary()
# %%
# Grab the seprate components
generator, discriminator = GAN.layers

# For every epcoh
for epoch in range(epochs):
    print(f"Currently on Epoch {epoch+1}")
    i = 0
    # For every batch in the dataset
    for X_batch in dataset:
        i=i+1
        if i%100 == 0:
            print(f"\tCurrently on batch number {i} of {len(my_data)//batch_size}")

        ## TRAINING THE DISCRIMINATOR ######
        
        # Create Noise
        noise = tf.random.normal(shape=[batch_size, codings_size])
        
        # Generate numbers based just on noise input
        gen_images = generator(noise)
        
        # Concatenate Generated Images against the Real Ones
        # TO use tf.concat, the data types must match!
        X_fake_vs_real = tf.concat([gen_images, tf.dtypes.cast(X_batch,tf.float32)], axis=0)
        
        # Targets set to zero for fake images and 1 for real images
        y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
        
        # This gets rid of a Keras warning
        discriminator.trainable = True
        
        # Train the discriminator on this batch
        discriminator.train_on_batch(X_fake_vs_real, y1)

        ## TRAINING THE GENERATOR     ######
        
        # Create some noise
        noise = tf.random.normal(shape=[batch_size, codings_size])
        
        # We want discriminator to belive that fake images are real
        y2 = tf.constant([[1.]] * batch_size)
        
        # Avois a warning
        discriminator.trainable = False
        
        GAN.train_on_batch(noise, y2)
        
print("TRAINING COMPLETE")            
#%%
noise = tf.random.normal(shape=[10, codings_size])
noise.shape
plt.imshow(noise)
# %%
image = generator(noise)
image.shape
# %%
plt.imshow(only_zeros[0])
# %%
plt.imshow(image[0])
# %%
