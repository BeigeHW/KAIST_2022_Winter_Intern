'''Probabilistic Matrix Factorization is done in order to reduce the time complexity of learning for collaborative
filtering system. Suppose that the original rating matrix is R in R(m*n). We could divide the rating matrix R into
(m*l), and (l*n) matrices, each matrices representing latent feature vectors for users and movies. Originally, we had
to learn m*n elements, but after the reconstruction we only need to learn l*(m+n) features. Since l is the
hyperparameter that the user set, it's a constant. Thus, this probabilistic algorithm scale linearly with the number of
observations and perform well on very sparse and imbalanced datasets.



'''


import pandas as pd
import torch

ratings = pd.read_csv('ml-latest-small/ratings.csv')
ratings.describe()



rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
n_users, n_movies = rating_matrix.shape
# Scaling ratings to between 0 and 1, this helps our model by constraining predictions
min_rating, max_rating = ratings['rating'].min(), ratings['rating'].max()
rating_matrix = (rating_matrix - min_rating) / (max_rating - min_rating)


# Replacing missing ratings with -1 so we can filter them out later
rating_matrix[rating_matrix.isnull()] = -1
rating_matrix = torch.FloatTensor(rating_matrix.values)



# This is how we can define our feature matrices
# We're going to be training these, so we'll need gradients
latent_vectors = 5
user_features = torch.randn(n_users, latent_vectors, requires_grad=True)
user_features.data.mul_(0.01)
movie_features = torch.randn(n_movies, latent_vectors, requires_grad=True)
movie_features.data.mul_(0.01)


class PMFLoss(torch.nn.Module):
    def __init__(self, lam_u=0.3, lam_v=0.3):
        super().__init__()
        self.lam_u = lam_u
        self.lam_v = lam_v

    def forward(self, matrix, u_features, v_features):
        non_zero_mask = (matrix != -1).type(torch.FloatTensor)
        predicted = torch.sigmoid(torch.mm(u_features, v_features.t()))

        diff = (matrix - predicted) ** 2
        prediction_error = torch.sum(diff * non_zero_mask)

        u_regularization = self.lam_u * torch.sum(u_features.norm(dim=1))
        v_regularization = self.lam_v * torch.sum(v_features.norm(dim=1))

        return prediction_error + u_regularization + v_regularization

criterion = PMFLoss()
loss = criterion(rating_matrix, user_features, movie_features)

# Actual training loop now

latent_vectors = 30
user_features = torch.randn(n_users, latent_vectors, requires_grad=True)
user_features.data.mul_(0.01)
movie_features = torch.randn(n_movies, latent_vectors, requires_grad=True)
movie_features.data.mul_(0.01)

pmferror = PMFLoss(lam_u=0.05, lam_v=0.05)
optimizer = torch.optim.Adam([user_features, movie_features], lr=0.01)
for step, epoch in enumerate(range(1000)):
    optimizer.zero_grad()
    loss = pmferror(rating_matrix, user_features, movie_features)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"Step {step}, {loss:.3f}")

# Checking if our model can reproduce the true user ratings
user_idx = 7
user_ratings = rating_matrix[user_idx, :]
true_ratings = user_ratings != -1
predictions = torch.sigmoid(torch.mm(user_features[user_idx, :].view(1, -1), movie_features.t()))
predicted_ratings = (predictions.squeeze()[true_ratings]*(max_rating - min_rating) + min_rating).round()
actual_ratings = (user_ratings[true_ratings]*(max_rating - min_rating) + min_rating).round()

print("Predictions: \n", predicted_ratings)
print("Truth: \n", actual_ratings)