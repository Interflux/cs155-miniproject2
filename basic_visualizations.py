"""
Filename:     basic_visualizations.py
Version:      1.1
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

# Import packages
import random


def main():

    # Load the movie data
    movie_data = np.loadtxt("data\\movies.txt", dtype="str", delimiter="\t")

    # Load the ratings data
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

    # For each movie...
    for movie_id in most_common_movies:

        # Get the title
        movie_title = movie_data[movie_id - 1][1]

        # Extract the ratings
        movie_ratings = np.asarray([x for x in ratings if x[1] == movie_id])
        
        # Count the frequency of each rating
        ratings_distribution = np.zeros(5, dtype="int")
        for rating in movie_ratings[:, 2]:
            ratings_distribution[rating - 1] += 1
        
        # Generate a histogram for the current most-commonly-rated movie as a bar chart
        plt.bar(np.arange(1, 5 + 1), ratings_distribution)
        plt.title("Ratings for the Top-" + str(n) + " Most-Commonly-Rated Movie: " + movie_title)
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.gca().grid(axis="y", linestyle="--")
        plt.savefig("distribution_for_top-" + str(n) + "_most_commonly_rated_movie_id-" + str(movie_id) + ".png", dpi=200)

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
    best_rated_movies = np.asarray(average_ratings[:10, 0], dtype="int")

    # For each movie...
    for movie_id in best_rated_movies:

        # Get the title
        movie_title = movie_data[movie_id - 1][1]

        # Extract the ratings
        movie_ratings = np.asarray([x for x in ratings if x[1] == movie_id])
    
        # Count the frequency of each rating
        ratings_distribution = np.zeros(5, dtype="int")
        for rating in movie_ratings[:, 2]:
            ratings_distribution[rating - 1] += 1
    
        # Generate a histogram of the best-rated movies as a bar chart
        plt.bar(np.arange(1, 5 + 1), ratings_distribution)
        plt.title("Ratings for the Top-" + str(n) + " Best-Rated Movie: " + movie_title)
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.gca().grid(axis="y", linestyle="--")
        plt.savefig("distribution_for_top-" + str(n) + "_highest_rated_movie_id-" + str(movie_id) + ".png", dpi=200)

        # Flush the plot
        plt.cla()
        plt.clf()
        plt.close()

    ##############################################################
    #                                                            #
    # 4. All ratings of movies from three genres of your choice. #
    #                                                            #
    ##############################################################

    # Set n for the number of genres to visualize
    n = 3
    
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
    
    # Select n genres to visualize
    genres = random.sample(range(2, 20 + 1), n)

    # For each genre, visualize the ratings for all the movies in the genre
    for genre in genres:
        # Identify all the movies in the genre
        genre_movies = []
        for movie in movie_data:
            if movie[genre] == "1":
                genre_movies.append(np.uint16(movie[0]))

        # Extract their ratings
        genre_ratings = np.asarray([x for x in ratings if x[1] in genre_movies])
    
        # Count the frequency of each rating
        genre_ratings_distribution = np.zeros(5, dtype="int")
        for rating in genre_ratings[:, 2]:
            genre_ratings_distribution[rating - 1] += 1
    
        # Generate a histogram of the genre-typed movies as a bar chart
        plt.bar(np.arange(1, 5 + 1), genre_ratings_distribution)
        plt.title("Distribution of Ratings for the Movies of Genre: \'" + genre_labels[genre] + "\'")
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.gca().grid(axis="y", linestyle="--")
        plt.savefig("distribution_of_" + genre_labels[genre] + "_movie_ratings.png", dpi=200)

        # Flush the plot
        plt.cla()
        plt.clf()
        plt.close()
    
    
if __name__ == "__main__":
    main()
