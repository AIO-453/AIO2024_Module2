import numpy as np


def compute_conditional_probability(train_data):
    y_unique = ['No', 'Yes']
    conditional_probability = []
    list_x_name = []

    prior_probability = np.zeros(len(y_unique))
    inter_P = np.zeros(len(y_unique))

    prior_probability[0] = (
        (train_data[:, len(train_data[0])-1] == "No").sum())/train_data.shape[0]
    prior_probability[1] = (
        (train_data[:, len(train_data[0])-1] == "Yes").sum())/train_data.shape[0]

    for i in range(0, train_data . shape[1] - 1):
        x_unique = np . unique(train_data[:, i])
        list_x_name . append(x_unique)

    # your code here ********************

    for i in range(len(list_x_name)):
        x_conditional_probability = []
        for j in list_x_name[i]:
            inter_P[0] = np.sum((train_data[:, 0] == j) & (
                train_data[:, -1] == y_unique[0]))/train_data.shape[0]
            inter_P[1] = np.sum((train_data[:, 0] == j) & (
                train_data[:, -1] == y_unique[1]))/train_data.shape[0]
            P_A_given_B = inter_P/prior_probability
            x_conditional_probability.append(P_A_given_B)
        conditional_probability . append(x_conditional_probability)
    return conditional_probability, list_x_name
