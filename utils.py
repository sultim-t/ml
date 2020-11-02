import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, vstack, coo_matrix
from sklearn.utils import shuffle
import random

def getData():
    #data_file_path = "./netflix-prize-data/processed_data.csv"
    #data_file_path = "./netflix-prize-data/processed_data_4m.csv"
    data_file_path = "netflix-prize-data/small.txt"

    df = pd.read_csv(data_file_path, header=None, names=['UserId', 'Rating', 'MovieId'])
    print(df.iloc[::5000000, :])

    instance_count, _ = df.shape

    # add column with 1
    ones_column = coo_matrix(np.full((instance_count, 1), 1))

    encoder = OneHotEncoder(categories='auto')

    # (number_of_ratings x number_of_users)
    one_hot_user_matrix = encoder.fit_transform(np.asarray(df['UserId']).reshape(-1, 1))
    print("One-hot user matrix shape: " + str(one_hot_user_matrix.shape))

    # (number_of_ratings x number_of_movie_ids)
    one_hot_movie_matrix = encoder.fit_transform(np.asarray(df['MovieId']).reshape(-1, 1))
    print("One-hot movie matrix shape: " + str(one_hot_movie_matrix.shape))

    # train data in CSR format
    X = hstack([ones_column, one_hot_user_matrix, one_hot_movie_matrix]).tocsr()
    # data to predict
    ratings = np.asarray(df['Rating']).reshape(-1, 1)

    return X, ratings

def getReadyData():
    X, ratings = getData()

    # do shuffling so records will be evenly distributed over the matrix
    X, ratings = shuffle(X, ratings)
    return X, ratings


def getRandomExcept(count, exceptThis):
    # inclusive
    a, b = exceptThis
    r = random.randint(0, count - (b + 1 - a) - 1)
    if r >= a:
        # if r in [a,..] then shift to skip [a,b]
        r += b + 1 - a
    return r

