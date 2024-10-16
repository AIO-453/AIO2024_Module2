# Import libraries
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(0)
# %matplotlib inline


def load_data_from_file(fileName='./advertising.csv'):
    data = np.genfromtxt(fileName, delimiter=',', skip_header=1)
    features_X = data[:, :-1]
    sale_y = data[:, -1]
    features_X = np.c_[np.ones((features_X.shape[0], 1)), features_X]

    return features_X, sale_y


def compute_loss(xdata, ydata, individual):
    N = len(ydata)
    theta = np.array(individual)
    y_pred = xdata.dot(theta)
    loss = (y_pred-ydata).T.dot(y_pred-ydata)/N

    return loss


def compute_fitness(xdata, ydata, individual):
    loss = compute_loss(xdata, ydata, individual)
    fitness_value = 1/(loss+1)

    return fitness_value


def create_individual(n=4, bound=10):
    individual = []
    for _ in range(n):
        individual.append(random.uniform(-bound/2, bound/2))
    return individual


def crossover(individual1, individual2, crossover_rate=0.9):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()

    for i in range(len(individual1)):
        if random.random() < crossover_rate:
            individual1_new[i] = individual2[i]
            individual2_new[i] = individual1[i]

    return individual1_new, individual2_new


def mutate(individual, mutation_rate=0.05):
    individual_m = individual.copy()
    n = len(individual)
    for i in range(n):
        if random.random() < mutation_rate:
            individual_m[i] = create_individual()[0]

    return individual_m


def selection(sorted_old_population, m=100):
    index1 = random.randint(0, m - 1)
    while True:
        index2 = random.randint(0, m - 1)
        if (index2 != index1):
            break
    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]

    return individual_s


def initializePopulation(m):
    population = np.array([create_individual() for _ in range(m)])
    return population


def create_new_population(xdata, ydata, old_population, elitism=2, gen=1):
    m = len(old_population)
    sorted_population = np.array(
        sorted(old_population, key=lambda ind: compute_fitness(xdata, ydata, ind)))

    if gen % 1 == 0:
        # print("Best loss:", compute_loss(xdata, ydata, sorted_population[m - 1]), "with chromosome:", sorted_population[m - 1])
        pass

    new_population = []

    while len(new_population) < m - elitism:
        # Selection
        parent1 = selection(sorted_population, m)
        parent2 = selection(sorted_population, m)

        # Crossover
        child1, child2 = crossover(parent1, parent2)

        # Mutation
        child1 = mutate(child1)
        child2 = mutate(child2)

        # Append new children to new population
        new_population.append(child1)
        new_population.append(child2)

    # Copy elitism chromosomes that have best fitness score to the next generation
    for ind in sorted_population[m - elitism:]:
        new_population.append(ind)

    return new_population, compute_loss(xdata, ydata, sorted_population[m - 1])


def run_GA():
    n_generations = 100
    m = 600
    features_X, sales_Y = load_data_from_file()
    # Initialize population with the correct feature size
    population = initializePopulation(m)
    losses_list = []

    for i in range(n_generations):
        # Tạo quần thể mới
        population, best_loss = create_new_population(
            features_X, sales_Y, population, elitism=2, gen=i+1)

        # Lưu lại giá trị loss tốt nhất của thế hệ hiện tại
        losses_list.append(best_loss)

        # Hiển thị loss của thế hệ hiện tại
        # print(f"Generation {i+1}: Best Loss = {best_loss}")

    return losses_list


def visualize_loss(losses_list):
    plt.figure(figsize=(10, 6))
    plt.plot(losses_list,  c='green', label="Loss")
    plt.title('Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_predict_gt(features_X, sales_Y, population):
    # Sắp xếp quần thể theo độ phù hợp (fitness) và chọn cá thể tốt nhất
    sorted_population = sorted(
        population, key=lambda ind: compute_fitness(features_X, sales_Y, ind))
    print("Best individual (theta):", sorted_population[-1])

    # Lấy trọng số (theta) của cá thể tốt nhất
    theta = np.array(sorted_population[-1])

    # Tính toán giá trị dự đoán (estimated_prices)
    estimated_prices = []
    for feature in features_X:
        # Tính dự đoán giá dựa trên theta và features_X
        estimated_price = np.dot(feature, theta)
        estimated_prices.append(estimated_price)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xlabel('Samples')
    plt.ylabel('Price')

    # Biểu đồ giá trị thực (sales_Y) và giá trị dự đoán (estimated_prices)
    plt.plot(sales_Y, c='green', label='Real Prices')
    plt.plot(estimated_prices, c='blue', label='Estimated Prices')

    plt.legend()
    plt.show()

# if __name__ == "__main__"
