"""
Filename:     advanced_visualizations.py
Version:      0.2
Date:         2018/2/21

Description:  Tests operations to generate advanced visualizations for
              CS 155's second miniproject.

Author(s):     Dennis Lam
Organization:  California Institute of Technology

"""

# Import packages as aliases
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

# Import packages
import scipy.linalg
import surprise


def svd_surprise(k=20, bias=True, test_fraction=0.25):
    """
    Performs SVD on the ratings data using surprise.

    """
    # Load the data
    reader = surprise.Reader(rating_scale=(1, 5), sep='\t')
    data = surprise.Dataset.load_from_file('data/data.txt', reader)

    # Split the data into a training set and test set
    train_set, test_set = surprise.model_selection.train_test_split(data, test_size=test_fraction)

    # Declare the model
    model = surprise.SVD(n_factors=k, biased=bias)

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
    mask = numpy.random.rand(len(df)) < 0.7
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

    # Cost function
    lr = tf.constant(.001, name='learning_rate')
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_step = optimizer.minimize(base_cost, global_step=global_step)

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
    df_test = df[~msk]
    user_indicies_test = [x-1 for x in df_test.user.values]
    item_indicies_test = [x-1 for x in df_test.item.values]
    rates_test = df_test.rate.values

    # Accuracy
    R_test = tf.gather(result_flatten, user_indicies_test * tf.shape(result)[1] + item_indicies_test, name='extracting_user_rate_test')
    diff_op_test = tf.subtract(R_test, rates_test, name='test_diff')
    diff_op_squared_test = tf.abs(diff_op, name="squared_difference_test")

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

# Perform SVD using Hamed's TensorFlow implementation
U, V = svd_hamed(k=20)

# Transpose U
U = np.matrix.transpose(U)

""" 
# Perform SVD using surprise
U, V, a, b = svd_surprise(k=20, bias=true)

# Transpose U and V
U = np.matrix.transpose(U)
V = np.matrix.transpose(V)

"""

### Project U and V onto 2 dimensions ###

# Run SVD on V, decomposing it into AS(B^T) where S is a diagonal matrix
A, S, B = scipy.linalg.svd(V)

# Save the first two columns of A
A_proj = A[:, 0:2]

# Project every movie and user using A_proj
A_proj_trans = np.matrix.transpose(A_proj)
V_proj = np.matmul(A_proj_trans, V)
U_proj = np.matmul(A_proj_trans, U)

### Visualize our results ###

# Make a crude plot of V_proj
plt.plot(V_proj[0], V_proj[1], 'ro')
plt.show()
