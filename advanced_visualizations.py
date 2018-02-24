"""
Filename:     advanced_visualizations.py
Version:      1.1
Date:         2018/2/23

Description:  Generates advanced visualizations for CS 155's second miniproject.

Author(s):     Garret Sullivan, Dennis Lam
Organization:  California Institute of Technology

"""

# Import package components
from collections import Counter

# Import packages as aliases
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

# Import packages
import scipy.linalg
import surprise
import operator
import random
import os


def svd_sgd(k=20, reg=0.0, eta=0.03, eps=0.0001, max_epochs=300, bias=False):
    """
    Performs SVD on the ratings data using SGD. Adapted from Homework 5 solutions.

    """
    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt').astype(int)
	
    m = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    n = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    
    U = np.random.random((m,k)) - 0.5
    V = np.random.random((k,n)) - 0.5
    A = np.random.random(m) - 0.5
    B = np.random.random(n) - 0.5
    
    size = Y_train.shape[0]
    delta = None
    indices = np.arange(size)    
    for epoch in range(max_epochs):
        # Run an epoch of SGD
        before_E_in = get_err_sgd(U, V, A, B, Y_train, reg, bias)
        np.random.shuffle(indices)
        for ind in indices:
            (i, j, Yij) = Y_train[ind]
            prediction = np.dot(U[i-1], V[:,j-1])
            
            # If the bias terms are enabled, add them to the prediction
            if bias:
                prediction = prediction + A[i-1] + B[j-1]
            
            # Update U[i], V[j]
            U[i-1] = U[i-1] - eta*(reg*U[i-1] - V[:,j-1]*(Yij - prediction))
            V[:,j-1] = V[:,j-1] - eta*(reg*V[:,j-1] - U[i-1]*(Yij - prediction))
            
            # If the bias terms are enabled, update them as well
            if bias:
                A[i-1] = A[i-1] - eta*(-(Yij - prediction))
                B[j-1] = B[j-1] - eta*(-(Yij - prediction))
            
        E_in = get_err_sgd(U, V, A, B, Y_train, reg, bias)

        # Compute change in E_in for first epoch
        if epoch == 0:
            delta = before_E_in - E_in

        # If E_in doesn't decrease by some fraction <eps>
        # of the initial decrease in E_in, stop early            
        elif before_E_in - E_in < eps * delta:
            break
        
    print("SVD Test RMSE: " + str(np.sqrt(get_err_sgd(U, V, A, B, Y_test, bias=bias))))
        
    return U, V


def get_err_sgd(U, V, A, B, Y, reg=0.0, bias=False):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T,
    possibly with bias terms.
    
    Adapted from Homework 5 solutions.
    
    """
    # Compute mean squared error on each data point in Y; include
    # regularization penalty in error calculations.
    # We first compute the total squared squared error
    err = 0.0
    for (i,j,Yij) in Y:
        prediction = np.dot(U[i-1], V[:,j-1])
        
        # If the bias terms are enabled, add them to the prediction
        if bias:
            prediction = prediction + A[i-1] + B[j-1]
        
        err += 0.5 *(Yij - prediction)**2
    # Add error penalty due to regularization if regularization
    # parameter is nonzero
    if reg != 0:
        U_frobenius_norm = np.linalg.norm(U, ord='fro')
        V_frobenius_norm = np.linalg.norm(V, ord='fro')
        err += 0.5 * reg * (U_frobenius_norm ** 2)
        err += 0.5 * reg * (V_frobenius_norm ** 2)
    # Return the mean of the regularized error
    return err / float(len(Y))


def svd_surprise(k=20, epochs=20, learning_rate=0.005, bias=True, test_fraction=0.0):
    """
    Performs SVD on the ratings data using surprise.

    """
    # Load the data
    reader = surprise.Reader(rating_scale=(1, 5), sep='\t')
    data = surprise.Dataset.load_from_file('data/data.txt', reader)

    if test_fraction == 0.0:
        _, test_set = surprise.model_selection.train_test_split(data, test_size=0.25)
        train_set = data.build_full_trainset()
    else:
        # Split the data into a training set and test set
        train_set, test_set = surprise.model_selection.train_test_split(data, test_size=test_fraction)

    # Declare the model
    model = surprise.SVD(n_factors=k, n_epochs=epochs, lr_all=learning_rate, biased=bias)

    # Train the model on the data
    model.fit(train_set)
    predictions = model.test(test_set)

    # Print the accuracy of the predictions
    print("SVD Test RMSE: " + str(surprise.accuracy.rmse(predictions, verbose=False)))

    # Return U, V, the user bias terms, and the movie bias terms
    return model.pu, model.qi, model.bu, model.bi


def svd_hamed(k=20):
    """
    Performs SVD on the ratings data using TensorFlow. Credit goes to Hamed Firooz:
    http://hameddaily.blogspot.com/2016/12/simple-matrix-factorization-with.html

    """
    # Read data
    df = pd.read_csv('data/data.txt', sep='\t', names=['user', 'item', 'rate'])
    mask = np.random.rand(len(df)) < 0.7
    df_train = df[mask]

    user_indicies = [x-1 for x in df_train.user.values]
    item_indicies = [x-1 for x in df_train.item.values]
    rates = df_train.rate.values

    # Variables
    feature_len = k
    U = tf.Variable(initial_value=tf.truncated_normal([943,feature_len]), name='users')
    P = tf.Variable(initial_value=tf.truncated_normal([feature_len,1682]), name='items')
    result = tf.matmul(U, P)
    result_flatten = tf.reshape(result, [-1])

    # Rating
    R = tf.gather(result_flatten, user_indicies * tf.shape(result)[1] + item_indicies, name='extracting_user_rate')

    # Cost function
    diff_op = tf.subtract(R, rates, name='training_diff')
    diff_op_squared = tf.abs(diff_op, name="squared_difference")
    base_cost = tf.reduce_sum(diff_op_squared, name="sum_squared_error")

    # Regularization
    lda = tf.constant(.001, name='lambda')
    norm_sums = tf.add(tf.reduce_sum(tf.abs(U, name='user_abs'), name='user_norm'),
                       tf.reduce_sum(tf.abs(P, name='item_abs'), name='item_norm'))
    regularizer = tf.multiply(norm_sums, lda, 'regularizer')
    cost = tf.add(base_cost, regularizer)

    # Cost function
    lr = tf.constant(.001, name='learning_rate')
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_step = optimizer.minimize(cost, global_step=global_step)

    # Execute
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(1000):
        sess.run(training_step)

    # Example: user=278 item=603 r=5 (from u.data)
    u, p, r = df[['user', 'item', 'rate']].values[0]
    rhat = tf.gather(tf.gather(result, u-1), p-1)
    print("rating for user " + str(u) + " for item " + str(p) + " is " + str(r) + " and our prediction is: " + str(sess.run(rhat)))

    # Calculate accuracy
    df_test = df[~mask]
    user_indicies_test = [x-1 for x in df_test.user.values]
    item_indicies_test = [x-1 for x in df_test.item.values]
    rates_test = df_test.rate.values

    # Accuracy
    R_test = tf.gather(result_flatten, user_indicies_test * tf.shape(result)[1] + item_indicies_test, name='extracting_user_rate_test')
    diff_op_test = tf.subtract(R_test, rates_test, name='test_diff')
    diff_op_squared_test = tf.abs(diff_op_test, name="squared_difference_test")

    cost_test = tf.divide(tf.reduce_sum(tf.square(diff_op_squared_test, name="squared_difference_test"), name="sum_squared_error_test"), df_test.shape[0] * 2., name="average_error")
    print(sess.run(cost_test))

    # Return latent vectors in NumPy format
    return U.eval(session=sess), P.eval(session=sess)


def print_ratings_dataframe(df):
    """
    Prints the dataframe containing the ratings data to standard output.

    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
        print(df)


def make_piazza_plots(V_proj, output_dir, n=30, movie_labels=True):
    """
    Generates hopefully interesting dataplots to share with the class on Piazza.

    """
    # Load the movie data
    movie_data = np.loadtxt("data\\movies.txt", dtype="str", delimiter="\t")

    # Load the movie ratings
    ratings = np.loadtxt("data\\data.txt", dtype="int")

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the lookup dictionary for the movie genres
    genre_labels = {2  : "Unknown",
                    3  : "Action",
                    4  : "Adventure",
                    5  : "Animation",
                    6  : "Childrens",
                    7  : "Comedy",
                    8  : "Crime",
                    9  : "Documentary",
                    10 : "Drama",
                    11 : "Fantasy",
                    12 : "Film-Noir",
                    13 : "Horror",
                    14 : "Musical",
                    15 : "Mystery",
                    16 : "Romance",
                    17 : "Sci-Fi",
                    18 : "Thriller",
                    19 : "War",
                    20 : "Western"}

    # Select the genres to visualize
    genres = np.arange(3, 20 + 1)

    # For each movie, count the number of ratings and add up their values
    total_ratings = {}
    for rating in ratings:
        if rating[1] in total_ratings:
            total_ratings[rating[1]][0] += rating[2]
            total_ratings[rating[1]][1] += 1
        else:
            total_ratings[rating[1]] = [rating[2], 1]
            
    # Compute the average rating for each movie
    average_ratings = {}
    for movie_id, rating_data in total_ratings.items():
        average_ratings[movie_id] = rating_data[0] / rating_data[1]
    
    # For each genre, visualize the ratings for the top-n most-commonly-rated movies in the genre
    for genre in genres:
        # Identify all the movies in the genre
        genre_movies = []
        for movie in movie_data:
            if movie[genre] == "1":
                genre_movies.append(np.uint16(movie[0]))

        # Identify the n movies with the greatest number of ratings

        freq_dict = {}

        for movie_id in genre_movies:
            freq_dict[movie_id] = 0
            
        for rating in ratings:
            if rating[1] in genre_movies:
                freq_dict[rating[1]] += 1

        id_list = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
        id_list = [x[0] for x in id_list[0:n]]
        
        # Plot the movies

        x_list = []
        y_list = []
        z_list = []

        for movie_id in id_list:
            x_list.append(V_proj[0][movie_id - 1])
            y_list.append(V_proj[1][movie_id - 1])
            z_list.append(average_ratings[movie_id])

        plt.figure(figsize=(12, 9))
        plt.scatter(x_list, y_list, c=z_list, s=100)
        plt.colorbar()
        
        if movie_labels:
            for i, movie_id in enumerate(id_list):
                plt.annotate(movie_data[movie_id - 1][1], xy=(x_list[i], y_list[i]), xycoords='data', xytext=(5, 5), fontsize=4, textcoords='offset points')

        plt.title("Most-Commonly-Rated \'" + genre_labels[genre] + "\' Movies")
        plt.savefig(output_dir + "\\scatter_top-" + str(n) + "_most_commonly_rated_" + genre_labels[genre] + "_movies.png", dpi=200)

        # Flush the plot
        plt.cla()
        plt.clf()
        plt.close()


def make_plots(V_proj, output_dir):
    """
    Generates a full suite of advanced visualizations from the given matrix.

    """
    # Load the movie data
    movie_data = np.loadtxt("data\\movies.txt", dtype="str", delimiter="\t")

    # Load the movie ratings
    ratings = np.loadtxt("data\\data.txt", dtype="int")

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ##################################################
    #                                                #
    # 2. All ratings of the ten most popular movies. #
    #                                                #
    ##################################################
        
    # Set n for the n most-commonly-rated movies to examine
    n = 10

    # Identify the n movies with the greatest number of ratings
    id_list = [list(x)[0] for x in Counter(ratings[:, 1]).most_common(n)]

    # Plot the movies

    x_list = []
    y_list = []

    for movie_id in id_list:
        x_list.append(V_proj[0][movie_id - 1])
        y_list.append(V_proj[1][movie_id - 1])

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(x_list, y_list)

    for i, movie_id in enumerate(id_list):
        ax.annotate(movie_data[movie_id - 1][1], xy=(x_list[i], y_list[i]), xycoords='data', xytext=(5, 5), textcoords='offset points')

    plt.title("Top-" + str(n) + " Most-Commonly-Rated Movies")
    plt.savefig(output_dir + "\\scatter_top-" + str(n) + "_most_commonly_rated_movies.png", dpi=80)

    # Flush the plot
    plt.cla()
    plt.clf()
    plt.close()

    ##########################################
    #                                        #
    # 3. All ratings of the ten best movies. #
    #                                        #
    ##########################################

    # Set n for the n best-rated movies to examine
    n = 10

    # For each movie, count the number of ratings and add up their values
    total_ratings = {}
    for rating in ratings:
        if rating[1] in total_ratings:
            total_ratings[rating[1]][0] += rating[2]
            total_ratings[rating[1]][1] += 1
        else:
            total_ratings[rating[1]] = [rating[2], 1]
            
    # Compute the average rating for each movie
    average_ratings = []
    for movie_id, rating_data in total_ratings.items():
        average_ratings.append([movie_id, rating_data[0] / rating_data[1]])
        
    # Sort the movies by average rating
    average_ratings = np.asarray(sorted(average_ratings, key = lambda x: x[1], reverse=True))

    # Identify the n best-rated movies
    id_list = np.asarray(average_ratings[:n, 0], dtype="int")

    # Plot the movies

    x_list = []
    y_list = []

    for movie_id in id_list:
        x_list.append(V_proj[0][movie_id - 1])
        y_list.append(V_proj[1][movie_id - 1])

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(x_list, y_list)

    for i, movie_id in enumerate(id_list):
        ax.annotate(movie_data[movie_id - 1][1], xy=(x_list[i], y_list[i]), xycoords='data', xytext=(5, 5), textcoords='offset points')

    plt.title("Top-" + str(n) + " Most-Highly-Rated Movies")
    plt.savefig(output_dir + "\\scatter_top-" + str(n) + "_most_highly_rated_movies.png", dpi=80)

    # Flush the plot
    plt.cla()
    plt.clf()
    plt.close()

    #####################################################
    #                                                   #  
    # All ratings of 10 movies from a particular genre. #
    #                                                   #
    #####################################################

    # Set k for the number of genres to visualize
    k = 19

    # Define the lookup dictionary for the movie genres
    genre_labels = {2  : "Unknown",
                    3  : "Action",
                    4  : "Adventure",
                    5  : "Animation",
                    6  : "Childrens",
                    7  : "Comedy",
                    8  : "Crime",
                    9  : "Documentary",
                    10 : "Drama",
                    11 : "Fantasy",
                    12 : "Film-Noir",
                    13 : "Horror",
                    14 : "Musical",
                    15 : "Mystery",
                    16 : "Romance",
                    17 : "Sci-Fi",
                    18 : "Thriller",
                    19 : "War",
                    20 : "Western"}

    # Select k genres to visualize
    genres = random.sample(range(2, 20 + 1), k)

    # Set n for the number of movies in each genre to visualize
    n = 10

    # For each genre, visualize the ratings for the top-n most-commonly-rated movies in the genre
    for genre in genres:
        # Identify all the movies in the genre
        genre_movies = []
        for movie in movie_data:
            if movie[genre] == "1":
                genre_movies.append(np.uint16(movie[0]))

        # Identify the n movies with the greatest number of ratings

        freq_dict = {}

        for movie_id in genre_movies:
            freq_dict[movie_id] = 0
            
        for rating in ratings:
            if rating[1] in genre_movies:
                freq_dict[rating[1]] += 1

        id_list = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
        id_list = [x[0] for x in id_list[0:n]]
        
        # Plot the movies

        x_list = []
        y_list = []

        for movie_id in id_list:
            x_list.append(V_proj[0][movie_id - 1])
            y_list.append(V_proj[1][movie_id - 1])

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.scatter(x_list, y_list)

        for i, movie_id in enumerate(id_list):
            ax.annotate(movie_data[movie_id - 1][1], xy=(x_list[i], y_list[i]), xycoords='data', xytext=(5, 5), textcoords='offset points')

        plt.title("Top-" + str(n) + " Most-Commonly-Rated \'" + genre_labels[genre] + "\' Movies")
        plt.savefig(output_dir + "\\scatter_top-" + str(n) + "_most_commonly_rated_" + genre_labels[genre] + "_movies.png", dpi=80)

        # Flush the plot
        plt.cla()
        plt.clf()
        plt.close()


def main():

    ##############################
    #                            #
    # TRAINING AND VISUALIZATION #
    #                            #
    ##############################

    """ If we're loading the data ourselves...

    # Load the movie data
    movies = np.loadtxt("data\\movies.txt", dtype="str", delimiter="\t")

    # Load the rating data
    ratings = np.loadtxt("data\\data.txt", dtype="int")

    # Stores the ratings in a pandas-readable dictionary
    ratings_dict = {'userID'  : [],
                    'movieID' : [],
                    'rating'  : []}

    # Populate the dictionary
    for rating in ratings:
        ratings_dict['userID'].append(rating[0])
        ratings_dict['movieID'].append(rating[1])
        ratings_dict['rating'].append(rating[2])

    # Load the dictionary into a dataframe
    df = pd.DataFrame(ratings_dict)

    """

    ############################################
    #                                          #
    # Choose a matrix-factorization algorithm. #
    #                                          #
    ############################################

    """ If we want to perform SVD using TensorFlow...

    # Perform SVD using Hamed's TensorFlow implementation
    U, V = svd_hamed(k=20)

    # Transpose U
    U = np.matrix.transpose(U)

    """

    """ If we want to perform SVD using our own SGD algorithm...

    # Perform SVD using SGD
    U, V = svd_sgd(k=20, reg=10**-1, bias=True)

    # Transpose U
    U = np.matrix.transpose(U)

    """

    """ If we want to perform SVD using surprise...

    # Perform SVD using surprise
    U, V, a, b = svd_surprise(k=20, bias=True)

    # Transpose U and V
    U = np.matrix.transpose(U)
    V = np.matrix.transpose(V)

    """

    #########################################
    #                                       #
    # Run a matrix-factorization algorithm. #
    #                                       #
    #########################################

    # Perform SVD using surprise
    U, V, a, b = svd_surprise(k=20, epochs=100, learning_rate=0.001, bias=True)

    # Transpose U and V
    U = np.matrix.transpose(U)
    V = np.matrix.transpose(V)

    ########################################
    #                                      #
    # Project U and V onto two dimensions. #
    #                                      #
    ########################################

    # Run SVD on V, decomposing it into AS(B^T) where S is a diagonal matrix
    A, S, B = scipy.linalg.svd(V)

    # Save the first two columns of A
    A_proj = A[:, 0:2]

    # Project every movie and user using A_proj
    A_proj_trans = np.matrix.transpose(A_proj)
    V_proj = np.matmul(A_proj_trans, V)
    U_proj = np.matmul(A_proj_trans, U)

    ##########################
    #                        #
    # Visualize our results! #
    #                        #
    ##########################

    # make_plots(V_proj, "svd_surprise")
    make_piazza_plots(V_proj, "piazza_plots\\surprise_k20")

    #################################
    #                               #
    # Repeat for other SGD methods. #
    #                               #
    #################################

    """
    
    ### TensorFlow... ###

    # Perform SVD using Hamed's TensorFlow implementation
    U, V = svd_hamed(k=20)

    # Transpose U
    U = np.matrix.transpose(U)

    # Run SVD on V, decomposing it into AS(B^T) where S is a diagonal matrix
    A, S, B = scipy.linalg.svd(V)

    # Save the first two columns of A
    A_proj = A[:, 0:2]

    # Project every movie and user using A_proj
    A_proj_trans = np.matrix.transpose(A_proj)
    V_proj = np.matmul(A_proj_trans, V)
    U_proj = np.matmul(A_proj_trans, U)

    # Plot the results
    make_plots(V_proj, "svd_tensorflow")

    ### Our own SGD algorithm... ###

    # Perform SVD using SGD
    U, V = svd_sgd(k=20, reg=10**-1, bias=True)

    # Transpose U
    U = np.matrix.transpose(U)

    # Run SVD on V, decomposing it into AS(B^T) where S is a diagonal matrix
    A, S, B = scipy.linalg.svd(V)

    # Save the first two columns of A
    A_proj = A[:, 0:2]

    # Project every movie and user using A_proj
    A_proj_trans = np.matrix.transpose(A_proj)
    V_proj = np.matmul(A_proj_trans, V)
    U_proj = np.matmul(A_proj_trans, U)

    # Plot the results
    make_plots(V_proj, "svd_sgd")

    ### Our own SGD algorithm, without bias... ###

    # Perform SVD using SGD
    U, V = svd_sgd(k=20, reg=10**-1, bias=False)

    # Transpose U
    U = np.matrix.transpose(U)

    # Run SVD on V, decomposing it into AS(B^T) where S is a diagonal matrix
    A, S, B = scipy.linalg.svd(V)

    # Save the first two columns of A
    A_proj = A[:, 0:2]

    # Project every movie and user using A_proj
    A_proj_trans = np.matrix.transpose(A_proj)
    V_proj = np.matmul(A_proj_trans, V)
    U_proj = np.matmul(A_proj_trans, U)

    # Plot the results
    make_plots(V_proj, "svd_sgd_nobias")

    """


if __name__ == "__main__":
    main()
