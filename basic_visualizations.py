"""
Filename:     basic_visualizations.py
Version:      0.9
Date:         2018/2/19

Description:  Generates basic visualizations for CS 155's second miniproject.

Author(s):     Dennis Lam
Organization:  California Institute of Technology

"""

# Import package components
from scipy import stats, integrate
from collections import Counter

# Import packages as aliases
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def main():

    # Load the movie ratings
    ratings = np.loadtxt("data\\data.txt", dtype="int")

    ############################################
    #                                          #
    # 1. All ratings in the MovieLens Dataset. #
    #                                          #
    ############################################
    
    # Count the frequency of each rating
    ratings_distribution = np.zeros(5, dtype="int")
    for rating in ratings[:, 2]:
        ratings_distribution[rating - 1] += 1

    # Generate a histogram of all movie ratings as a bar chart
    plt.bar(np.arange(1, 5 + 1), ratings_distribution)
    plt.title("Distribution of All Movie Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.gca().grid(axis="y", linestyle="--")
    plt.savefig("distribution_of_all_movie_ratings.png", dpi=200)

    # Flush the plot
    plt.cla()
    plt.clf()
    plt.close()

    ##################################################
    #                                                #
    # 2. All ratings of the ten most popular movies. #
    #                                                #
    ##################################################
    
    # Set n for the n most-commonly-rated movies to examine
    n = 10
    
    # Identify the n movies with the greatest number of ratings
    most_common_movies = [list(x)[0] for x in Counter(ratings[:, 1]).most_common(n)]

    # Extract their ratings
    most_common_ratings = np.asarray([x for x in ratings if x[1] in most_common_movies])
    
    # Count the frequency of each rating
    most_common_ratings_distribution = np.zeros(5, dtype="int")
    for rating in most_common_ratings[:, 2]:
        most_common_ratings_distribution[rating - 1] += 1
    
    # Generate a histogram of the most-commonly-rated movies as a bar chart
    plt.bar(np.arange(1, 5 + 1), most_common_ratings_distribution)
    plt.title("Distribution of Ratings for the " + str(n) + " Most-Commonly-Rated Movies")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.gca().grid(axis="y", linestyle="--")
    plt.savefig("distribution_of_most_commonly_rated_movie_ratings.png", dpi=200)

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
    best_rated_movies = average_ratings[:10, 0]

    # Extract their ratings
    best_ratings = np.asarray([x for x in ratings if x[1] in best_rated_movies])
    
    # Count the frequency of each rating
    best_ratings_distribution = np.zeros(5, dtype="int")
    for rating in best_ratings[:, 2]:
        best_ratings_distribution[rating - 1] += 1
    
    # Generate a histogram of the best-rated movies as a bar chart
    plt.bar(np.arange(1, 5 + 1), best_ratings_distribution)
    plt.title("Distribution of Ratings for the " + str(n) + " Best-Rated Movies")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.gca().grid(axis="y", linestyle="--")
    plt.savefig("distribution_of_highest_rated_movie_ratings.png", dpi=200)

    # Flush the plot
    plt.cla()
    plt.clf()
    plt.close()

    ##############################################################
    #                                                            #
    # 4. All ratings of movies from three genres of your choice. #
    #                                                            #
    ##############################################################

    # TODO
    
    
if __name__ == "__main__":
    main()
