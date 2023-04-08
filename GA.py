from base import *
def create_initial_population(cities: List[City], clusters: List[List[City]], pop_size: int) -> List[List[City]]:
    population = []
    for _ in range(pop_size):
        cluster_indices = list(range(len(clusters)))
        random.shuffle(cluster_indices)
        individual = []
        for cluster_index in cluster_indices:
            cluster = clusters[cluster_index]
            individual.extend(cluster)
        population.append(individual)
    return population

def compute_total_distance(cities: List[City]) -> float:
    total_distance = 0
    num_cities = len(cities)
    for i in range(num_cities-1):
        current_city = cities[i]
        next_city = cities[(i + 1) % num_cities]
        total_distance += euclidean_distance(current_city, next_city)
    return total_distance

def rank_individuals(cities: List[City], population: List[List[City]]) -> List[Tuple[float, List[City]]]:
    fitness_results = []
    for individual in population:
        fitness_results.append((compute_total_distance(individual), individual))
    return sorted(fitness_results, key=lambda x: x[0])

def selection(population_ranked: List[Tuple[float, City]], elite_size: int) -> List[List[int]]:
    selection_results = [ind[1] for ind in population_ranked[:elite_size]]
    return selection_results

def crossover(parent1: List[City], parent2: List[City]) -> Tuple[List[City], List[City]]:
    # Create child arrays filled with -1
    child1 = [-1] * len(parent1)
    child2 = [-1] * len(parent2)

    # Choose random start and end indices for the crossover section
    start_index, end_index = sorted(random.sample(range(len(parent1)), 2))
    # Copy the crossover section from parent1 to child1 and from parent2 to child2
    child1[start_index:end_index] = parent1[start_index:end_index]
    child2[start_index:end_index] = parent2[start_index:end_index]

    # Fill the remaining elements in the children by iterating through the parents' elements in a circular manner
    for child, parent in [(child1, parent2), (child2, parent1)]:
        index = end_index
        for value in parent:
            if value not in child:
                while child[index % len(child)] != -1:
                    index += 1
                child[index % len(child)] = value

    return child1, child2

def cycle_crossover(parent1: List[City], parent2: List[City]) -> Tuple[List[City], List[City]]:
    # Choose a random starting point
    index = random.randint(0,len(parent1)-1)

    parents = [parent1,parent2]
    parnet_index = -1
    child = [-1] * len(parent1)

    # Cycle through the parents and create a child
    while -1 in child:
        parnet_index = (parnet_index + 1) % 2

        while child[index] != -1:
            index += 1
        
        while child[index] == -1:
            child[index] = parents[parnet_index][index]
            index = parents[(parnet_index + 1) % 2].index(child[index])
    
    return child

def mutate(individual: List[City], mutation_rate: float) -> List[City]:
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = random.randint(0, len(individual) - 1)
            individual[i], individual[swap_with] = individual[swap_with], individual[i]
    return individual

def breed_population(mating_pool: List[List[City]], elite_individuals: List[List[City]], elite_size: int, mutation_rate: float) -> List[List[City]]:
    offspring = elite_individuals
    for i in range(elite_size, len(mating_pool)):
        child1, child2 = crossover(mating_pool[i], mating_pool[len(mating_pool) - i - 1])
        #child = cycle_crossover(mating_pool[i], mating_pool[len(mating_pool) - i - 1])
        offspring.append(mutate(child2, mutation_rate))
    return offspring

def genetic_algorithm(cities: List[City], clusters: List[List[City]], pop_size: int, elite_size: int, mutation_rate: float, generations: int) -> Tuple[float, List[City]]:
    population = create_initial_population(cities, clusters, pop_size)
    best_individual = None
    best_distance = float('inf')

    print("유전 알고리즘 시작")
    for i in tqdm(range(generations)):
        population_ranked = rank_individuals(cities, population)
        if i % 10 == 0:
            print(f"Generation {i + 1}")
            print("Best: ", population_ranked[0][0])
            print("Worst: ", population_ranked[-1][0])
        #population = selection(population_ranked, POP_SIZE)
        #elite_individuals = population[:ELITE_SIZE]
        elite_individuals = selection(population_ranked, ELITE_SIZE)
        offspring = breed_population(population, elite_individuals, elite_size, mutation_rate)
        population = offspring
        current_best_distance, current_best_individual = population_ranked[0]
        #print(f"Generation {i + 1} Best distance: {current_best_distance}")
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_individual = current_best_individual
            print(f"Generation {i + 1}: Best distance: {best_distance}")

    return best_distance, best_individual