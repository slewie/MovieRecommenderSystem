import pandas as pd
from surprise import dump, Reader, Dataset
import numpy as np
import random


class Recommender:
    """
   Recommendation system for movies recommendations.

   This class allows to:
       - Load user ratings and movies metadata
       - Train recommendation models for personalized suggestions
       - Generate top-n movie recommendations for a user
       - Continuously update predictions based on new ratings
       - Add new users on the fly
       - Give information about the best films and user preferences

   The recommendation logic works as follows:
       - For new users - returns most popular unwatched movies
       - For users with some ratings:
           - Trains KNN model on available ratings
           - Predicts scores for unwatched movies per user
           - Returns top-n movies by predicted rating

       - Handles new ratings by:
           - Updating internal userId-to-movieId mappings
           - Retraining the model periodically when new data accumulates

   Key attributes:
       - model: trained KNN recommender
       - ratings: dataset with user ratings
       - movies_info: movies metadata sets
   """

    def __init__(self, ratings_file_name, movies_file_name, model_file_name, genres_file_name, max_iter=10,
                 random_seed=0):
        """
        Initialize the recommender

        :param ratings_file_name: path to the ratings dataset
        :param movies_file_name: path to the movies info dataset
        :param model_file_name: path to the pre-trained model
        :param max_iter: max iterations without model retraining
        :param random_seed: random seed for reproducibility
        """
        self.data = pd.read_csv(ratings_file_name, index_col=0)[["user", "item", "rating"]]
        self.movies_info = pd.read_csv(movies_file_name, index_col=0)
        self.genres = pd.read_csv(genres_file_name, index_col=0)
        self.users = set(self.data.user.values)
        self.movies = set(self.data.item.values)
        self.user_genres = {}
        self.user_movies = {}
        self.movies_views = np.zeros(max(self.movies) + 1)
        self._init_info_matrices()

        reader = Reader(rating_scale=(1, 5))
        self.ratings = Dataset.load_from_df(self.data, reader)

        _, self.model = dump.load(model_file_name)
        self.iter_without_training = 0
        self.max_iter = max_iter

        self._set_seed(random_seed)

    def _init_info_matrices(self):
        """Initialize matrices for tracking information"""
        for user in self.users:
            user_q = self.data.query(f'user == {user}')
            self.user_movies[user] = set(user_q.item)

            user_q_best_movies = list(user_q.query(f'rating >= 4').item)
            self.user_genres[user] = np.sum(
                self.movies_info.query(f'movieId in {user_q_best_movies}').iloc[:, 4:].values, axis=0)

        for movie in self.movies:
            movie_views = self.data.query(f'item == {movie}').shape[0]
            self.movies_views[movie] = movie_views

    @staticmethod
    def _set_seed(seed):
        """Set random number generator seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)

    def _train_model(self):
        """Retrain recommendation model on the full dataset"""
        self.model.fit(self.ratings.build_full_trainset())
        self.iter_without_training = 0

    def get_recs(self, user_id, n=5):
        """
        Get top-n movie recommendations for user.
        If the user has rated few movies, he will receive the most popular movies as a recommendation.

        :param user_id: target user id
        :param n: number of recommendations

        :return: DataFrame with recommended movie ids and titles
        """
        if user_id not in self.users:
            raise ValueError(f"User with id = {user_id} doesn't exist")

        if len(self.user_movies[user_id]) < self.max_iter:
            sorted_movies = np.argsort(self.movies_views)[::-1]
            new_user_movies = self.movies - self.user_movies[user_id]
            movies_ids = [movie for movie in sorted_movies if movie in new_user_movies][:n]
        else:
            if len(self.user_movies[user_id]) == self.max_iter or self.iter_without_training == self.max_iter:
                self._train_model()
            new_user_movies = [movie for movie in self.movies if movie not in self.user_movies[user_id]]
            user_ratings = []
            for movie in new_user_movies:
                user_ratings.append((movie, self.model.predict(user_id, movie).est))
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            movies_ids = [i[0] for i in user_ratings[:n]]
        self.iter_without_training += 1
        return pd.DataFrame(
            data={
                'movieId': movies_ids,
                'movieTitle':
                    [self.movies_info.query(f'movieId == {movie_id}').movieTitle.values[0] for movie_id in movies_ids]
            }
        )

    def add_rating(self, user_id, item_id, rating):
        """
        Add new rating from user and updates the information
        :param user_id: user id
        :param item_id: movie id
        :param rating: rating value
        """
        if user_id not in self.users:
            raise ValueError(f"User with id = {user_id} doesn't exist")

        if item_id not in self.movies:
            raise ValueError(f"Item with id = {item_id} doesn't exist")

        if rating not in list(range(1, 6)):
            raise ValueError(f"Rating should be in range from 1 to 5")

        if item_id in self.user_movies[user_id]:
            raise ValueError(f"The user has already watched this movie")

        self.ratings.raw_ratings.append((user_id, item_id, rating, None))
        self.user_movies[user_id].add(item_id)
        self.movies_views[item_id] += 1
        if rating >= 4:
            self.user_genres[user_id] += self.movies_info.query(f'movieId == {item_id}').iloc[:, 4:].values.reshape(-1)

    def add_user(self, user_id):
        """
        Add new user to the system
        :param user_id: new user id
        """
        if user_id in self.users:
            raise ValueError(f"User with id = {user_id} already exists")

        self.users.add(user_id)
        self.user_movies[user_id] = set()
        self.user_genres[user_id] = np.zeros(18)

    def get_most_popular_films(self, n=5):
        """
        Return dataframe with top-n most globally popular movies

        :param n: number of popular movies to return

        :return: DataFrame with popular movie ids, titles, views
        """
        movies_ids = np.argsort(self.movies_views)[::-1][:n]
        movie_views = self.movies_views[movies_ids]
        return pd.DataFrame(
            data={
                'movieId': movies_ids,
                'movieTitle':
                    [self.movies_info.query(f'movieId == {movie_id}').movieTitle.values[0] for movie_id in movies_ids],
                'viewsNum': movie_views.astype(int)
            }
        )

    def get_user_best_films(self, user_id, n=5):
        """
        Return dataframe with user_id's top-n the highest rated movies

        :param user_id: target user id
        :param n: number of the best movies to return

        :return: DataFrame with the best movie ids, titles, ratings
        """
        user_q = self.data.query(f'user == {user_id}').sort_values(by='rating', ascending=False)
        movies_ids = user_q.item.values[:n]
        ratings = user_q.rating.values[:n]
        return pd.DataFrame(
            data={
                'movieId': movies_ids,
                'movieTitle':
                    [self.movies_info.query(f'movieId == {movie_id}').movieTitle.values[0] for movie_id in movies_ids],
                'rating': ratings.astype(int)
            }
        )

    def get_user_most_popular_genres(self, user_id, n=5):
        """
        Return dataframe with top-n genre preferences for target user

        :param user_id: target user id
        :param n: number of top genres to return

        :return: DataFrame with genre ids, names and rating counts
        """
        genres_ids = np.argsort(self.user_genres[user_id])[::-1][:n]
        genre_num = self.user_genres[user_id][genres_ids]
        genres_ids += 1
        return pd.DataFrame(
            data={
                'genreId': genres_ids,
                'genreName':
                    [self.genres.query(f'genreId == {genre_id}').genreName.values[0] for genre_id in genres_ids],
                'moviesNum': genre_num.astype(int)
            }
        )
