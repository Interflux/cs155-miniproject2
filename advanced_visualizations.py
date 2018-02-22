"""
Filename:     advanced_visualizations.py
Version:      0.1
Date:         2018/2/21

Description:  Performs operations to generate advanced visualizations for
              CS 155's second miniproject.

Author(s):     Dennis Lam
Organization:  California Institute of Technology

"""

# Import packages as aliases
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import packages
import surprise
import scipy


def main():

    """
    INITIAL COMMIT: Just getting familiar with the surprise package and making
    sure we can actually use its functions to perform the algorithms we want
    on the data!

    """

    """ First, load the data """
    
    # Load the movie data
    movies = np.loadtxt("data\\movies.txt", dtype="str", delimiter="\t")

    # Load the rating data
    ratings = np.loadtxt("data\\data.txt", dtype="int")

    """ Then, convert the data to a dataframe so it can be read by surprise """

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
        
    # Declare a reader with the appropriate rating scale
    reader = surprise.Reader(rating_scale=(1, 5))

    # Load the dataframe into a surprise-readable data structure
    data = surprise.Dataset.load_from_df(df[['userID', 'movieID', 'rating']], reader)

    """ Finally, let's run a SVD algorithm on the data """

    # Split the data into a training set and test set
    train_set, test_set = surprise.model_selection.train_test_split(data, test_size=0.25)

    # Set the number of factors
    k = 20
    
    # Declare the model
    model = surprise.SVD(n_factors=k, biased=False)

    # Train the model on the data
    model.fit(train_set)
    predictions = model.test(test_set)
    
    # Print the accuracy of the predictions
    print("Unbiased-SVD Test RMSE: " + str(surprise.accuracy.rmse(predictions, verbose=False)))

    """ Now let's run an SVD algorithm with bias terms on the data """

    # Declare the model
    model = surprise.SVD(n_factors=k, biased=True)

    # Train the model on the data
    model.fit(train_set)
    predictions = model.test(test_set)
    
    # Print the accuracy of the predictions
    print("Biased-SVD Test RMSE: " + str(surprise.accuracy.rmse(predictions, verbose=False)))


if __name__ == "__main__":
    main()
