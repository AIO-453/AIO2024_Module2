import Genetic_Algorithm as ga

if __name__ == "__main__":
    features_X, sales_Y = ga.load_data_from_file(fileName='./advertising.csv')
    individual = ga.create_individual()

    print(sales_Y.shape)
    print(individual)

    individual = [4.09, 4.82, 3.10, 4.02]
    fitness_score = ga.compute_fitness(features_X, sales_Y, individual)
    print(fitness_score)

    individual1 = [4.09, 4.82, 3.10, 4.02]
    individual2 = [3.44, 2.57, -0.79, -2.41]
    individual1, individual2 = ga.crossover(individual1, individual2, 2.0)
    print(" individual1 : ", individual1)
    print(" individual2 : ", individual2)

    before_individual = [4.09, 4.82, 3.10, 4.02]
    after_individual = ga.mutate(individual, mutation_rate=2.0)
    print(before_individual == after_individual)

    individual1 = [4.09, 4.82, 3.10, 4.02]
    individual2 = [3.44, 2.57, -0.79, -2.41]
    old_population = [individual1, individual2]
    new_population, _ = ga.create_new_population(
        features_X, sales_Y, old_population, elitism=2, gen=1)

    # Chạy thuật toán GA và visual hóa loss
    losses_list = ga.run_GA()
    ga.visualize_loss(losses_list)
    ga.visualize_predict_gt(features_X, sales_Y, old_population)
