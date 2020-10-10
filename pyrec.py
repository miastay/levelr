import numpy as np
import pandas as pd
#import cv2
import sys
import csv
import os
import math
#import os.path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Reshape, Dot
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l2

def RecommenderV1(n_users, n_tracks, n_factors):
    user = Input(shape=(1,))
    u = Embedding(n_users, n_factors, embeddings_initializer='he_normal',
                  embeddings_regularizer=l2(1e-6))(user)
    u = Reshape((n_factors,))(u)
    
    track = Input(shape=(1,))
    t = Embedding(n_tracks, n_factors, embeddings_initializer='he_normal',
                  embeddings_regularizer=l2(1e-6))(track)
    t = Reshape((n_factors,))(t)
    
    x = Dot(axes=1)([u, t])
    model = Model(inputs=[user, track], outputs=x)
    opt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


path = sys.argv[1] + ''

#lines = data.splitlines()
#reader = csv.reader(lines)
#data = list(reader)
#os.chdir('C:/Users/ryan')
data = pd.read_csv(path,delimiter = ',')

#print(data)

ratings = data['plays']*(data['saved']+1)
ratings *= -0.8
ratings += 4
ratings = np.exp(ratings)
ratings += 1
ratings = 1/ratings

user_enc = LabelEncoder()
data['user'] = user_enc.fit_transform(data['user_id'].values)
n_users = data['user'].nunique()
item_enc = LabelEncoder()
data['track'] = item_enc.fit_transform(data['track_id'].values)
n_tracks = data['track'].nunique()
ratings = ratings.values.astype(np.float32)
min_rating = min(ratings)
max_rating = max(ratings)
#print(n_users, "....", n_tracks, "....", min_rating, "....", max_rating)

X = data[['user', 'track']].values
y = ratings
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape, "....", X_test.shape, "....", y_train.shape, "....", y_test.shape)

n_factors = 50 ##replace
X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]

model = RecommenderV1(n_users, n_tracks, n_factors)
history = model.fit(x=X_train_array, y=y_train, batch_size=2, epochs=5,
                    verbose=1, validation_data=(X_test_array, y_test))

#ratings = 1 / (1 + math.exp(4 - 0.8*(x)))

#print(ratings)