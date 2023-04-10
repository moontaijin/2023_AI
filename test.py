import numpy as np
import random
import pandas as pd
import heapq
import json
import csv

from typing import List, Tuple
from math import sqrt
from tqdm import tqdm
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        x_dis = abs(self.x - city.x)
        y_dis = abs(self.y - city.y)
        distance = np.sqrt((x_dis ** 2) + (y_dis ** 2))
        return distance

    def __repr__(self):
        return f"({self.x}, {self.y})"

def generate_cities(data_path,city_size):
    city = pd.read_csv(data_path,header=None,names=['x','y'],nrows=city_size)
    return [City(row['x'], row['y']) for index, row in city.iterrows()]

def euclidean_distance(city1: City, city2: City) -> float:
    return sqrt((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2)

with open('config.json','r',encoding='utf-8') as f:
    load_data = json.load(f)

CITY_SIZE = load_data['CITY_SIZE']
CITY_LIST = generate_cities(load_data['CITY_LIST_PATH'], CITY_SIZE)
NUM_CLUSTER = load_data['NUM_CLUSTER']
POP_SIZE = load_data['POP_SIZE']
ELITE_SIZE = load_data['ELITE_SIZE']
MUTATION_RATE = load_data['MUTATION_RATE']
GENERATIONS = load_data['GENERATIONS']

def cluster_cities(cities, num_clusters):
    city_coordinates = np.array([[city.x, city.y] for city in cities])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(city_coordinates)
    clusters = {i: [] for i in range(num_clusters)}
    for city, cluster_label in zip(cities, kmeans.labels_):
        clusters[cluster_label].append(city)
    return list(clusters.values())

def solve_subproblems(clusters):
    solutions = []
    print("클러스터 최적해 찾기")
    for cluster in tqdm(clusters):
        paths = []
        if (len(cluster) < 4):
            paths.append([cluster[0], cluster[0], cluster[0], cluster[0]])
            
            for _ in range(6):
                paths.append(cluster)

        else:
            direction = []

            cluster.sort(key=lambda city: city.x)
            direction.append(cluster[0])            # left

            k = len(cluster) - 1
            while True:
                if cluster[k] in direction:
                    k = k - 1
                else:
                    direction.append(cluster[k])    # right
                    break

            cluster.sort(key=lambda city: city.y)
            k = 0
            while True:
                if cluster[k] in direction:
                    k = k + 1
                else:
                    direction.append(cluster[k])    # down
                    break

            k = len(cluster) - 1
            while True:
                if cluster[k] in direction:
                    k = k - 1
                else:
                    direction.append(cluster[k])    # up
                    break


            paths.append(direction)
            paths.append(greedy(cluster, direction[3], direction[0]))               #1
            paths.append(greedy(cluster, direction[3], direction[2]))               #2
            paths.append(greedy(cluster, direction[3], direction[1]))               #3
            paths.append(greedy(cluster, direction[0], direction[2]))               #4
            paths.append(greedy(cluster, direction[0], direction[1]))               #5
            paths.append(greedy(cluster, direction[2], direction[1]))               #6

        solutions.append(paths)

    return solutions

def greedy(cities, start_city, end_city):
    city_indices = list(range(len(cities)))
    start_city_index = cities.index(start_city)
    end_city_index = cities.index(end_city)
    visited = [city_indices.pop(start_city_index)]  # 시작 도시 인덱스를 방문 목록에 추가하고 목록에서 제거
    
    city_indices.remove(end_city_index)
    while city_indices:
        # 현재 도시에서 가장 가까운 도시 인덱스를 찾음
        closest_city_index = min(city_indices, key=lambda index: cities[visited[-1]].distance(cities[index]))
        visited.append(closest_city_index)  # 가장 가까운 도시 인덱스를 방문 목록에 추가
        city_indices.remove(closest_city_index)  # 가장 가까운 도시 인덱스를 목록에서 제거
    visited.append(end_city_index)

    path = []
    for node in visited:
        path.append(cities[visited[node]])

    return path

def create_initial_cluster_population(cities: List[City], clusters: List[List[List[City]]], pop_size: int) -> List[List[List[City]]]:
    population = []

    cluster_indices = list(range(len(clusters)))
    visited = [cluster_indices.pop(0)]
    while cluster_indices:
        closest_cluster_index = min(cluster_indices, key=lambda index: clusters[visited[-1]][0][0].distance(clusters[index][0][0]))
        visited.append(closest_cluster_index)
        cluster_indices.remove(closest_cluster_index)

    individual = []
    for cluster_index in visited:
        cluster = clusters[cluster_index]
        individual.append(cluster)
    population.append(individual)

    for _ in range(pop_size - 1):
        cluster_indices = list(range(len(clusters)))
        random.shuffle(cluster_indices)
        individual = []
        for cluster_index in cluster_indices:
            cluster = clusters[cluster_index]
            individual.append(cluster)
        population.append(individual)
    return population

def extend_individual(individual: List[List[City]]) -> List[City]:
    extended_individual = []
    for cluster in individual:
        extended_individual.extend(cluster)

    return extended_individual

def compute_total_distance(cities: List[City]) -> float:
    total_distance = 0
    num_cities = len(cities)
    for i in range(num_cities-1):
        current_city = cities[i]
        next_city = cities[(i + 1) % num_cities]
        total_distance += euclidean_distance(current_city, next_city)

    return total_distance

def rank_individuals(cities: List[City], population: List[List[List[List[City]]]]) -> List[Tuple[float, List[City]]]:
    fitness_results = []

    for individual in population:
        e = -1
        s = -1
        k = -1
        distance = -1

        for i in range(4):
            for j in range(4):
                if s == -1 or individual[len(individual) - 1][0][i].distance(individual[0][0][j]) < distance:
                    e = i
                    s = j
                    k = j
                    distance = individual[len(individual) - 1][0][i].distance(individual[0][0][j])

        real_path = []

        for t in range(len(individual)):
            distance = -1
            a = -1
            b = -1

            if t == len(individual) - 1:
                a = e
                b = s
            else:
                for i in range(4):
                    for j in range(4):
                        if i != k and (distance == -1 or individual[t][0][i].distance(individual[t + 1][0][j]) < distance):
                            a = i
                            b = j
            
            # up left down right

            if k == 0:
                if a == 1:
                    real_path.append(individual[t][1])
                elif a == 2:
                    real_path.append(individual[t][2])
                elif a == 3:
                    real_path.append(individual[t][3])
            elif k == 1:
                if a == 0:
                    real_path.append(list(reversed(individual[t][1])))
                elif a == 2:
                    real_path.append(individual[t][4])
                elif a == 3:
                    real_path.append(individual[t][5])
            elif k == 2:
                if a == 0:
                    real_path.append(list(reversed(individual[t][2])))
                elif a == 1:
                    real_path.append(list(reversed(individual[t][4])))
                elif a == 3:
                    real_path.append(individual[t][6])
            else:
                if a == 0:
                    real_path.append(list(reversed(individual[t][3])))
                elif a == 1:
                    real_path.append(list(reversed(individual[t][5])))
                elif a == 2:
                    real_path.append(list(reversed(individual[t][6])))

            k = b

        extended_individual = extend_individual(real_path)
        fitness_results.append((compute_total_distance(extended_individual), individual, extended_individual))

    return sorted(fitness_results, key=lambda x: x[0])

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

def mutate(individual: List[List[City]], mutation_rate: float) -> List[City]:
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = random.randint(0, len(individual) - 1)
            individual[i], individual[swap_with] = individual[swap_with], individual[i]
    return individual

def selection(population_ranked: List[Tuple[float, List[List[List[City]]]]], elite_size: int) -> List[List[int]]:
    selection_results = [ind[1] for ind in population_ranked[:elite_size]]
    return selection_results

def breed_population(mating_pool: List[List[List[List[City]]]], elite_individuals: List[List[List[City]]], elite_size: int, mutation_rate: float) -> List[List[City]]:
    offspring = elite_individuals
    for i in range(elite_size, len(mating_pool)):
        child1, child2 = crossover(mating_pool[i], mating_pool[len(mating_pool) - i - 1])
        #child = cycle_crossover(mating_pool[i], mating_pool[len(mating_pool) - i - 1])
        offspring.append(mutate(child2, mutation_rate))
    return offspring

def genetic_algorithm(cities: List[City], clusters: List[List[List[City]]], pop_size: int, elite_size: int, mutation_rate: float, generations: int) -> Tuple[float, List[City]]:
    population = create_initial_cluster_population(cities, clusters, pop_size)
    best_individual = None
    best_path = None
    best_distance = float('inf')

    print("유전 알고리즘 시작")
    for i in tqdm(range(generations)):
        population_ranked = rank_individuals(cities, population)

        if i % 10 == 0:
            print(f"Generation {i + 1}")
            print("Best: ", population_ranked[0][0])
            print("Worst: ", population_ranked[-1][0])
        elite_individuals = selection(population_ranked, ELITE_SIZE)
        offspring = breed_population(population, elite_individuals, elite_size, mutation_rate)
        population = offspring
        current_best_distance, current_best_individual, current_real_path = population_ranked[0]
        #print(f"Generation {i + 1} Best distance: {current_best_distance}")
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_individual = current_best_individual
            best_path = current_real_path
            print(f"Generation {i + 1}: Best distance: {best_distance}")

    return best_distance, best_path

def test():
    # K-means 클러스터링으로 도시들을 클러스터로 분할
    clusters = cluster_cities(CITY_LIST, NUM_CLUSTER)

    # 각 클러스터에 대해 최적해를 찾음 (트리 탐색 사용)
    subproblem_solutions = solve_subproblems(clusters)

    best_distance, best_order = genetic_algorithm(CITY_LIST, subproblem_solutions, POP_SIZE, ELITE_SIZE, MUTATION_RATE, GENERATIONS)

    # 경로 시각화
    plt.figure(figsize=(12, 8))
    for i, city in enumerate(best_order):
        plt.scatter(city.x, city.y, c='blue', edgecolors='k', s=2)
        next_city = best_order[i + 1] if i + 1 < len(best_order) else best_order[0]
        plt.plot([city.x, next_city.x], [city.y, next_city.y],c='skyblue',alpha=0.5)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Best TSP Route (Total Distance: {best_distance:.2f})")
    plt.savefig(f'results/test_CL{NUM_CLUSTER}_POP{POP_SIZE}_GEN{GENERATIONS}.png')

    # 결과 저장
    with open(f'results/test_CL{NUM_CLUSTER}_POP{POP_SIZE}_GEN{GENERATIONS}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for city in best_order:
            writer.writerow([city.x, city.y])

if __name__ == '__main__':
    test()