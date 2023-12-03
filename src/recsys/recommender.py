import pandas as pd
from surprise import dump, Reader, Dataset
import numpy as np


class Recommender:

    def __init__(self, ratings_file_name, movies_file_name, model_file_name):
        # TODO: save info to the file
        self.data = pd.read_csv(ratings_file_name, index_col=0)[["user", "item", "rating"]]
        self.movies_info = pd.read_csv(movies_file_name, index_col=0)
        self.users = set(self.data.user.values)
        self.movies = set(self.data.item.values)
        self.user_genres = {}
        self.user_movies = {}
        self.movies_views = np.zeros(max(self.movies) + 1)
        self._init_info_matrices()

        reader = Reader(rating_scale=(1, 5))
        self.ratings = Dataset.load_from_df(self.data, reader)

        _, self.model = dump.load(model_file_name)

    def _init_info_matrices(self):
        for user in self.users:
            user_q = self.data.query(f'user == {user}')
            self.user_movies[user] = set(user_q.item)

            user_q_best_movies = list(user_q.query(f'rating >= 4').item)
            self.user_genres[user] = np.sum(
                self.movies_info.query(f'movieId in {user_q_best_movies}').iloc[:, 4:].values, axis=0)

        for movie in self.movies:
            movie_views = self.data.query(f'item == {movie}').shape[0]
            self.movies_views[movie] = movie_views

    def get_recs(self, user_id, n=5):
        if user_id not in self.users:
            raise ValueError(f"User with id = {user_id} doesn't exist")
        if len(self.user_movies[user_id]) < 10:
            sorted_movies = np.argsort(self.movies_views)[::-1]
            new_user_movies = self.movies - self.user_movies[user_id]
            movies_ids = [movie for movie in sorted_movies if movie in new_user_movies][:n]
        else:
            new_user_movies = [movie for movie in self.movies if movie not in self.user_movies[user_id]]
            user_ratings = []
            for movie in new_user_movies:
                user_ratings.append((movie, self.model.predict(user_id, movie).est))
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            movies_ids = [i[0] for i in user_ratings[:n]]
        return pd.DataFrame(
            data={
                'movieId': movies_ids,
                'movieTitle':
                    self.movies_info.query(f'movieId in {list(movies_ids)}').movieTitle.values
            }
        )

    def add_rating(self, user_id, item_id, rating):
        if user_id not in self.users:
            raise ValueError(f"User with id = {user_id} doesn't exist")

        if item_id not in self.movies:
            raise ValueError(f"Item with id = {item_id} doesn't exist")

        if rating not in list(range(1, 6)):
            raise ValueError(f"Rating should be in range from 1 to 5")
        self.ratings.raw_ratings.append((user_id, item_id, rating, None))
        self.user_movies[user_id].add(item_id)
        self.movies_views[item_id] += 1
        if rating >= 4:
            self.user_genres[user_id] += self.movies_info.query(f'movieId == {item_id}').iloc[:, 4:].values.reshape(-1)

    def add_user(self, user_id):
        if user_id in self.users:
            raise ValueError(f"User with id = {user_id} already exists")

        self.users.add(user_id)
        self.user_movies[user_id] = set()
        self.user_genres[user_id] = np.zeros(19)
