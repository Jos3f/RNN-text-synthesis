import numpy as np
import matplotlib.pyplot as plt
import copy
import json
from datetime import datetime
from tqdm import tqdm
import pickle
from pathlib import Path
import csv


class RNN:
    ind_to_char = None
    char_to_ind = None
    one_hot = None

    weights = None
    gradients = None
    h = None
    training_loss = 0

    ada_m = None

    train_info = None
    iter_vs_loss = None

    def __init__(self, book_data_all, node_count=100):
        book_data = []
        list(map(book_data.extend, book_data_all))
        # book_data = sum(book_data_all, [])
        self.ind_to_char = list(set(book_data))
        self.char_to_ind = {}
        for ind, char in enumerate(self.ind_to_char): self.char_to_ind[char] = ind
        self.one_hot = np.identity(len(self.ind_to_char))
        self.ada_m = {}
        self.initialiseWeights(node_count)
        self.train_info = []
        return

    def initialiseWeights(self, node_count):
        self.weights = {}
        self.weights["b"] = np.zeros((node_count, 1))
        self.weights["c"] = np.zeros((len(self.ind_to_char), 1))

        sig = 0.01
        self.weights["U"] = np.random.normal(size=(node_count, len(self.ind_to_char))) * sig
        self.weights["W"] = np.random.normal(size=(node_count, node_count)) * sig
        self.weights["V"] = np.random.normal(size=(len(self.ind_to_char), node_count)) * sig

        self.gradients = {}
        self.gradients["b"] = np.zeros_like(self.weights["b"])
        self.gradients["c"] = np.zeros_like(self.weights["c"])
        self.gradients["U"] = np.zeros_like(self.weights["U"])
        self.gradients["W"] = np.zeros_like(self.weights["W"])
        self.gradients["V"] = np.zeros_like(self.weights["V"])

        self.reset()
        return

    def reset(self):
        self.h = np.zeros((self.weights["W"].shape[1], 1))

    def softmax(self, X):
        return np.exp(X) / np.sum(np.exp(X), axis=0)

    def generate_text(self, length, start_char=".", upsample=False):
        return_string = start_char
        x = np.array([self.one_hot[self.char_to_ind[start_char]]]).T
        h = self.h.copy()

        # Forward pass
        for i in range(length):
            a = self.weights["W"] @ h + self.weights["U"] @ x + self.weights["b"]
            h = np.tanh(a)
            o = self.weights["V"] @ h + self.weights["c"]
            p = self.softmax(o)

            if upsample:
                temp_p = p.copy()
                p = p ** 1.25
                p = p / np.sum(p)

            cprobs = np.cumsum(p)

            a = np.random.uniform(0, 1, 1)
            ixs = np.argwhere(cprobs - a > 0)
            if len(ixs) == 0:  # Can be the case because of rare floating point error
                ii = np.argmax(p)
            else:
                ii = ixs[0][0]

            return_string += self.ind_to_char[ii]
            x = self.one_hot[ii:ii + 1].T

        return return_string

    def loss(self, Y, P):
        row_wise_dot = np.einsum('ij, ij->i', Y.T, P.T)
        loss = - np.sum(np.log(row_wise_dot + 1e-16))
        return loss

    def forward_pass(self, X):
        P = np.zeros((self.weights["V"].shape[0], X.shape[1]))
        H = np.zeros((X.shape[1], self.h.shape[0]))
        A = np.zeros((X.shape[1], self.weights["W"].shape[0]))

        for i in range(X.shape[1]):
            H[i:i + 1, :] = self.h.T
            a = self.weights["W"] @ self.h + self.weights["U"] @ X[:, i:i + 1] + self.weights["b"]
            self.h = np.tanh(a)
            o = self.weights["V"] @ self.h + self.weights["c"]
            P[:, i:i + 1] = self.softmax(o)
            A[i:i + 1, :] = a.T
        return P, H, A

    def update_gradients(self, X, Y):
        n = X.shape[1]
        P, H, A = self.forward_pass(X)
        loss = self.loss(Y, P) / n
        if self.training_loss == 0:
            self.training_loss = loss
        else:
            self.training_loss = .999 * self.training_loss + 0.001 * loss

        o_grad = -(Y - P).T
        G = o_grad

        H_shifted = H.copy()[1:, :]
        H_shifted = np.vstack((H_shifted, self.h.T))
        self.gradients["V"] = np.zeros_like(self.gradients["V"])
        self.gradients["V"] = G.T @ H_shifted
        '''for i in range(H_shifted.shape[0]):
            self.gradients["V"] += G[i:i+1,:].T @ H_shifted[i:i+1,:]'''
        self.gradients["c"] = G.T @ np.ones((n, 1))

        h_grad = np.zeros((n, self.h.shape[0]))
        h_grad[n - 1:n, :] = o_grad[n - 1:n, :] @ self.weights["V"]
        a_grad = np.zeros_like(A)
        for t in range(n - 1, 0, -1):
            a_grad[t:t + 1, :] = h_grad[t:t + 1, :] @ np.diag(1 - (H_shifted[t, :].T) ** 2)
            h_grad[t - 1:t, :] = o_grad[t - 1:t, :] @ self.weights["V"] + a_grad[t:t + 1, :] @ self.weights["W"]

        a_grad[0:1] = h_grad[0:1, :] @ np.diag(1 - (H_shifted[0, :].T) ** 2)

        G = a_grad

        self.gradients["W"] = G.T @ H
        self.gradients["U"] = G.T @ X.T
        self.gradients["b"] = G.T @ np.ones((n, 1))

        return

    def update_weights(self, eta):
        """
        AdaGrad
        """
        for weight in self.weights:
            clipped_gradients = np.clip(self.gradients[weight], -5, 5)
            if weight not in self.ada_m:
                self.ada_m[weight] = clipped_gradients ** 2
            else:
                self.ada_m[weight] += clipped_gradients ** 2

            self.weights[weight] -= eta * clipped_gradients / ((self.ada_m[weight] + np.finfo(float).eps) ** 0.5)

    def train(self, book_data_all, epochs=5, seq_length=25, eta=0.1, generate_each=500, progress_bars=True):
        self.train_info.append({'epochs': epochs, "seq_length": seq_length, 'eta': eta})

        self.reset()
        best_weights = copy.deepcopy(self.weights)
        lowest_loss = np.inf
        book_data_ind = []
        for book_data in book_data_all:
            book_data_ind.append([self.char_to_ind[char] for char in book_data])

        loss_iter = []
        loss_val = []

        update_step = 0
        for ep in range(epochs):
            print("\nStarting epoch {}".format(ep + 1))
            for book_data_i in range(len(book_data_all)):
                self.reset()
                u_steps = (range(0, len(book_data_all[book_data_i]) - 2, seq_length))
                if progress_bars:
                    u_steps = tqdm(u_steps)

                # for e in tqdm(range(0, len(book_data_all[book_data_i]) - seq_length - 1, seq_length)):
                for e in u_steps:
                    update_step += 1
                    X_chars = book_data_ind[book_data_i][e:min(e + seq_length, len(book_data_all[book_data_i]) - 1)]
                    Y_chars = book_data_ind[book_data_i][e + 1:min(e + 1 + seq_length, len(book_data_all[book_data_i]))]

                    X = self.one_hot[:, X_chars]
                    Y = self.one_hot[:, Y_chars]

                    self.update_gradients(X, Y)
                    self.update_weights(eta)
                    if update_step % generate_each == 0:
                        print("\nUpdate step: {}, Loss: {}".format(update_step, self.training_loss))
                        print(self.generate_text(200, self.ind_to_char[Y_chars[-1]]))
                        print("Upsampled: ")
                        print(self.generate_text(200, self.ind_to_char[Y_chars[-1]], upsample=True))
                    loss_iter.append(update_step)
                    loss_val.append(self.training_loss)
                    if self.training_loss < lowest_loss:
                        lowest_loss = self.training_loss
                        best_weights = copy.deepcopy(self.weights)

            self.weights = best_weights

        iter_vs_loss = np.vstack((np.array(loss_iter), (np.array(loss_val))))
        if self.iter_vs_loss is None:
            self.iter_vs_loss = iter_vs_loss
        else:
            self.iter_vs_loss = np.hstack((self.iter_vs_loss, iter_vs_loss))

        return self.iter_vs_loss

    def save_weights(self, model_name):
        write_dir = 'models/{}/'.format(model_name)
        Path(write_dir).mkdir(parents=True, exist_ok=True)
        for key in self.weights:
            np.savetxt(write_dir + key + '.csv', self.weights[key], delimiter=',')

        wtr = csv.writer(open(write_dir + 'ind_to_char.csv', 'w'), delimiter=',', lineterminator='\n')
        for x in self.ind_to_char: wtr.writerow([x])

    def num_grads(self, X, Y, h=1e-4):
        grads = copy.deepcopy(self.gradients)
        original_weights = copy.deepcopy(self.weights)
        for weight in self.weights:
            for i in range(self.weights[weight].shape[0]):
                for j in range(self.weights[weight].shape[1]):
                    self.weights[weight][i, j] -= h
                    self.reset()
                    P = self.forward_pass(X)[0]
                    l1 = self.loss(Y, P)

                    self.weights[weight][i, j] += 2 * h
                    self.reset()
                    P = self.forward_pass(X)[0]
                    l2 = self.loss(Y, P)

                    grads[weight][i, j] = (l2 - l1) / (2 * h)
                    self.weights[weight][i, j] -= h

        self.weights = copy.deepcopy(original_weights)
        return grads


def check_gradients(book_data):
    h = 1e-4
    m = 5
    rnn = RNN(book_data, node_count=m)
    seq_length = 25

    book_data = book_data[0]
    book_data_ind = [rnn.char_to_ind[char] for char in book_data[:seq_length + 1]]
    X_chars = book_data_ind[:seq_length]
    Y_chars = book_data_ind[1:]
    X = rnn.one_hot[:, X_chars]
    Y = rnn.one_hot[:, Y_chars]

    rnn.update_gradients(X, Y)
    grads = rnn.gradients
    num_grads = rnn.num_grads(X, Y, h=h)
    print("Gradient check using the relative difference. seq_length={}, m={}".format(seq_length, m))

    max_diff_all = 0
    eps = 1e-16
    print("Weights & max & avg \\\\")
    for k in grads:
        diff = np.abs(grads[k] - num_grads[k])
        denom = (np.abs(grads[k]) + np.abs(num_grads[k])).clip(eps)
        diff = diff / denom
        max_diff = np.max(diff)
        if max_diff > max_diff_all:
            max_diff_all = max_diff
        print("{} & {} & {} \\\\".format(k, max_diff, np.mean(diff)))

    print("Max all &  {} & \\\\".format(max_diff_all))

    pass


def harry_potter_book():
    """
    Train on Harry Potter book and synthesise some text.
    :return:
    """
    # Read data
    bookfname = "datasets/goblet_book.txt"
    remove_chars = []
    # remove_chars = ['\n', '\t']
    book_data = [[ch for ch in open(bookfname, encoding="utf-8").read() if ch not in remove_chars]]

    # Check gradients
    check_gradients(book_data)

    # Create network
    rnn = RNN(book_data, node_count=100)
    generated_text = rnn.generate_text(200)
    print("Initially generated text: {}".format(generated_text))

    # Start training
    iter_vs_loss = rnn.train(book_data, epochs=100, generate_each=10000, seq_length=25)

    # Plot the loss
    plt.plot(iter_vs_loss[0], iter_vs_loss[1], label="Training smooth loss (avg)")
    plt.title("Loss per update step.")
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Generate some text after training
    print("\nGenerate for model with lowest loss ({}):".format(np.min(iter_vs_loss[1])))
    print(rnn.generate_text(1000, start_char=' '))

    print("Upsampled: ")
    print(rnn.generate_text(1000, start_char=' ', upsample=True))


    rnn.save_weights('model_2')

    print("Done")
    return


def trump_tweets():
    model_dir = "models/"
    model_name = "model_1"
    model_file = model_dir + model_name + ".pkl"
    save_model = True

    # read data
    data_dir = "datasets/trump_tweets/"
    # years = np.arange(2009, 2019) all years
    years = np.arange(2015, 2019)  # only data from 2015-2018
    tweets = []
    for year in years:
        print("Loading year {}".format(year))
        file = "{}condensed_{}.json".format(data_dir, year)
        with open(file, encoding='utf-8') as f:
            data = json.load(f)
            for tweet in data:
                tweet_formatted = [char for char in tweet["text"]]
                tweet_formatted.append('¤')
                tweets.append(tweet_formatted)

    # Only 2020 data from https://www.kaggle.com/austinreese/trump-tweets/data#realdonaldtrump.csv
    '''tweets = []
    reader = csv.DictReader(open(data_dir + "realdonaldtrump.csv", encoding="utf-8"))
    for tweet in reader:
        if tweet["date"][:4] == "2020":
            tweets.append([char for char in tweet["content"]])'''

    # Check if model have been saved
    try:
        with open(model_file, 'rb') as input:
            print("Loading saved model...")
            rnn = pickle.load(input)

            # Train more:
            # iter_vs_loss = rnn.train(tweets, epochs=5, generate_each=2000, seq_length=25, progress_bars=False, eta=0.1)

            iter_vs_loss = rnn.iter_vs_loss
    except:
        print("Model not found. Training new model...")

        rnn = RNN(tweets, node_count=100)
        generated_text = rnn.generate_text(200)
        print("Initially generated text: {}".format(generated_text))

        # seq_length=5 may work
        iter_vs_loss = rnn.train(tweets, epochs=10, generate_each=2000, seq_length=25, progress_bars=False)

    if save_model:
        with open(model_file, 'wb') as output:
            print("Saving model to {}.".format(model_file))
            pickle.dump(rnn, output, pickle.HIGHEST_PROTOCOL)

    # Plot the loss
    # plt.plot(iter_vs_loss[0], iter_vs_loss[1], label="Training smooth loss")
    plt.plot(iter_vs_loss[1], label="Training smooth loss (avg)")
    plt.title("Mean Loss per update step.")
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Generate some text after training
    print("\nGenerate for model with lowest loss ({}):".format(np.min(iter_vs_loss[1])))
    rnn.reset()
    for i in range(0, 100):
        print("\nTweet {}".format(i))
        rnn.reset()
        random_index = np.random.randint(len(tweets))
        start_char = tweets[random_index][0]
        print(rnn.generate_text(140, start_char, upsample=True))

    print("Done")
    return


def main():
    harry_potter_book()
    # trump_tweets()


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
