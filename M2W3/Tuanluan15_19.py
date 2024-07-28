# 15b
import numpy as np
import pandas as pd

# Defining the dataset
data = {
    'Day': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Overcast', 'Sunny', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

# Creating the DataFrame
df = pd.DataFrame(data)


def create_train_data(Data_Frame):
    r = Data_Frame.to_numpy()
    r = r[:, 1:]
    return r


train_data = create_train_data(df)
print(train_data)
print()


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


def get_index_from_value(feature_name, list_features):
    return np . where(list_features == feature_name)[0][0]


# ###################
# Prediction
# ###################
def prediction_play_tennis(X, list_x_name, prior_probability, conditional_probability):

    p0 = 0
    p1 = 0

    x1 = get_index_from_value(X[0], list_x_name[0])
    x2 = get_index_from_value(X[1], list_x_name[1])
    x3 = get_index_from_value(X[2], list_x_name[2])
    x4 = get_index_from_value(X[3], list_x_name[3])

    # P(outlook='Sunny', temperature='Cool', humidity='High', wind='Strong'| play_tennis='No')

    # P(outlook='Sunny'| play_tennis='No')*P(temperature='Cool'| play_tennis='No')*P(humidity='High'| play_tennis='No')*P(wind='Strong'| play_tennis='No')*P(play_tennis='No')
    P_X_given_No = conditional_probability[0][x1][0]*conditional_probability[1][x2][0] * \
        conditional_probability[2][x3][0] * \
        conditional_probability[3][x4][0]*prior_probability[0]

    # P(outlook='Sunny', temperature='Cool', humidity='High', wind='Strong'| play_tennis='Yes')
    P_X_given_Yes = conditional_probability[0][x1][1]*conditional_probability[1][x2][1] * \
        conditional_probability[2][x3][1] * \
        conditional_probability[3][x4][1]*prior_probability[1]
    P_X = P_X_given_No + P_X_given_Yes
    p0 = P_X_given_No/(P_X+1e-6)
    p1 = P_X_given_Yes/(P_X+1e-6)

    if p0 > p1:
        y_pred = ("Ad should go!")
    else:
        y_pred = ("Ad should go!")
    return y_pred


if __name__ == '__main__':
    _, list_x_name = compute_conditional_probability(train_data)

    print("x1 = ", list_x_name[0])
    print("x2 = ", list_x_name[1])
    print("x3 = ", list_x_name[2])
    print("x4 = ", list_x_name[3])
    print()

    # 16c
    outlook = list_x_name[0]
    i1 = get_index_from_value("Overcast", outlook)
    i2 = get_index_from_value("Rain", outlook)
    i3 = get_index_from_value("Sunny", outlook)
    print(i1, i2, i3)
    print()

    # 17d
    conditional_probability, list_x_name = compute_conditional_probability(
        train_data)
    # Compute P(" Outlook "=" Sunny "| Play Tennis "=" Yes ")
    x1 = get_index_from_value("Sunny", list_x_name[0])

    print("P( ' Outlook '= ' Sunny '| Play Tennis '= 'Yes ') = ",
          np . round(conditional_probability[0][x1][1], 2))
    print()

    # 18a
    conditional_probability, list_x_name = compute_conditional_probability(
        train_data)
    # Compute P(" Outlook "=" Sunny "| Play Tennis "=" yes ")
    x1 = get_index_from_value("Sunny", list_x_name[0])
    print("P( ' Outlook '= ' Sunny '| Play Tennis '= 'No') = ",
          np . round(conditional_probability[0][x1][0], 2))
    print()

    # 19b
    y_unique = ['No', 'Yes']
    prior_probability = np.zeros(len(y_unique))
    prior_probability[0] = (
        (train_data[:, len(train_data[0])-1] == "No").sum())/train_data.shape[0]
    prior_probability[1] = (
        (train_data[:, len(train_data[0])-1] == "Yes").sum())/train_data.shape[0]

    X = ['Sunny', 'Cool', 'High', 'Strong']
    conditional_probability, list_x_name = compute_conditional_probability(
        train_data)

    Prediction = prediction_play_tennis(X, list_x_name, prior_probability,
                                        conditional_probability)

    print(f'Prediction: {Prediction}')
