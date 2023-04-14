from base import *
def create_initial_population(cities: List[City], clusters: List[List[City]], pop_size: int) -> List[List[City]]:
    population = []
    # pop_size 만큼의 individual을 가진 population 생성
    for _ in range(pop_size):
        # 인덱스 리스트 생성
        cluster_indices = list(range(len(clusters)))
        # 인덱스 리스트 랜덤 셔플
        random.shuffle(cluster_indices)
        individual = []
        for cluster_index in cluster_indices:
            # 랜덤 셔플된 인덱스 순서대로 클러스터 순서 변경
            cluster = clusters[cluster_index]
            individual.extend(cluster)
        population.append(individual)
    return population

def create_initial_cluster_population(cities: List[City], clusters: List[List[City]], pop_size: int) -> List[List[List[City]]]:
    population = []
    # pop_size 만큼의 individual을 가진 population 생성
    for _ in range(pop_size):
        # 인덱스 리스트 생성
        cluster_indices = list(range(len(clusters)))
        # 인덱스 리스트 랜덤 셔플
        random.shuffle(cluster_indices)
        individual = []
        for cluster_index in cluster_indices:
            # 랜덤 셔플된 인덱스 순서대로 클러스터 순서 변경
            cluster = clusters[cluster_index]
            individual.append(cluster)
        population.append(individual)
    return population

def create_population(cities: List[City], pop_size: int) -> List[List[City]]:
    population = []
    # pop_size 만큼의 individual을 가진 population 생성
    for _ in range(pop_size):
        population.append(cities)
    return population

def compute_total_distance(cities: List[City]) -> float:
    total_distance = 0
    # 노드의 개수 산출
    num_cities = len(cities)
    for i in range(num_cities):
        # 노드와 다음 노드 사이의 유클리디안 거리를 산출해 total_distance에 반영
        current_city = cities[i]
        next_city = cities[(i + 1) % num_cities]
        total_distance += euclidean_distance(current_city, next_city)

    return total_distance

def extend_individual(individual: List[List[City]]) -> List[City]:
    # 클러스터 단위로 묶여있는 City 리스트를 하나의 City 리스트로 연결
    extended_individual = []
    for cluster in individual:
        extended_individual.extend(cluster)

    return extended_individual

def rank_individuals(cities: List[City], population: List[List[City]]) -> List[Tuple[float, List[City]]]:
    fitness_results = []
    # 각각 individual들의 Total Distance를 계산
    for individual in population:
        # 클러스터 단위로 묶여있는 City 리스트를 하나로 이음
        extended_individual = extend_individual(individual)
        # total_distance 계산
        fitness_results.append((compute_total_distance(extended_individual), individual))
    # 내림차순으로 정렬
    return sorted(fitness_results, key=lambda x: x[0])

def rank_populations(cities: List[City], population: List[List[City]]) -> List[Tuple[float, List[City]]]:
    fitness_results = []
    # 각각 individual들의 Total Distance를 계산
    for individual in population:
        fitness_results.append((compute_total_distance(individual), individual))
    # 내림차순으로 정렬
    return sorted(fitness_results, key=lambda x: x[0])

def selection(population_ranked: List[Tuple[float, City]], elite_size: int) -> List[List[int]]:
    # population_ranked에서 상위 elite_size만큼을 선택
    selection_results = [ind[1] for ind in population_ranked[:elite_size]]
    return selection_results

def crossover(parent1: List[City], parent2: List[City]) -> Tuple[List[City], List[City]]:
    # -1로 채워진 자식 배열 생성
    child1 = [-1] * len(parent1)
    child2 = [-1] * len(parent2)

    # 랜덤 시작, 끝 인덱스 생성
    start_index, end_index = sorted(random.sample(range(len(parent1)), 2))
    # parent1에서 child1로, parent2에서 child2로 복사
    child1[start_index:end_index] = parent1[start_index:end_index]
    child2[start_index:end_index] = parent2[start_index:end_index]

    # 부모의 요소를 원형으로 자식의 나머지 요소를 채움
    for child, parent in [(child1, parent2), (child2, parent1)]:
        index = end_index
        for value in parent:
            if value not in child:
                while child[index % len(child)] != -1:
                    index += 1
                child[index % len(child)] = value

    return child1, child2

def cycle_crossover(parent1, parent2):
    # 랜덤하게 시작 인덱스 선택
    start_index = random.randint(0, len(parent1) - 1)
    child = [-1] * len(parent1)
    # cycle crossover 적용
    while True:
        # 현재 인덱스의 값이 이전에 방문한 값인 경우, 사이클이 형성된 것으로 판단
        if child[start_index] != -1:
            break
        # 자식 개체에 부모1의 값 할당
        child[start_index] = parent1[start_index]
        # 부모2의 값으로 인덱스 찾아 이동
        index = parent2.index(parent1[start_index])
        start_index = index
    # 자식 개체에 부모2의 값 할당
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = parent2[i]
    return child

def mutate(individual: List[List[City]], mutation_rate: float) -> List[List[City]]:
    for i in range(len(individual)):
        # mutate_rate 보다 작은 값이 나오면 돌연변이 적용
        if random.random() < mutation_rate:
            # 랜덤하게 뽑은 두개의 클러스터의 인덱스를 변경
            swap_with = random.randint(0, len(individual) - 1)
            individual[i], individual[swap_with] = individual[swap_with], individual[i]
    
    return individual

def breed_population(mating_pool: List[List[City]], elite_individuals: List[List[City]], elite_size: int, mutation_rate: float) -> List[List[City]]:
    # 상위 elite_size만큼의 individual들은 그대로 다음 세대로 전달
    offspring = elite_individuals
    # pop_size에서 이미 다음세대로 전달된 elite_size를 뺀 횟수만큼 교배
    for i in range(elite_size, len(mating_pool)):
        # pop_size에서 elite_size를 제외한 나머지 individual들을 양 끝에서부터 순서대로 매칭시켜 교배
        child1, child2 = crossover(mating_pool[i], mating_pool[len(mating_pool) - i - 1])
        #child = cycle_crossover(mating_pool[i], mating_pool[len(mating_pool) - i - 1])

        # 돌연변이 적용
        #offspring.append(mutate(child, mutation_rate))
        offspring.append(mutate(child2, mutation_rate))
    return offspring

def genetic_algorithm(cities: List[City], clusters: List[List[City]], pop_size: int, elite_size: int, mutation_rate: float, generations: int) -> Tuple[float, List[City]]:
    # 초기 population 생성
    #population = create_initial_population(cities, clusters, pop_size)
    population = create_initial_cluster_population(cities, clusters, pop_size)
    #population = create_population(clusters, pop_size)

    # 초기 최고해, 최단 거리 설정
    best_individual = None
    best_distance = float('inf')

    print("유전 알고리즘 시작")
    for i in tqdm(range(generations)):
        # 100세대마다 결과 저장
        if i >0 and i%100 == 0:
            best_order = extend_individual(best_individual)
            os.mkdir(f"results/{EXP_NAME}/GEN{i}")
            plt.figure(figsize=(12, 8))
            for j, city in enumerate(best_order):
                plt.scatter(city.x, city.y, c='blue', edgecolors='k', s=2)
                next_city = best_order[j + 1] if j + 1 < len(best_order) else best_order[0]
                plt.plot([city.x, next_city.x], [city.y, next_city.y],c='skyblue',alpha=0.5)
            
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"Best TSP Route (Total Distance: {best_distance:.2f})")
            plt.savefig(f'results/{EXP_NAME}/GEN{i}/test_CL{NUM_CLUSTER}_POP{POP_SIZE}.png')

            with open(f'results/{EXP_NAME}/GEN{i}/test_CL{NUM_CLUSTER}_POP{POP_SIZE}.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for city in best_order:
                    writer.writerow([city.x, city.y])

        # population을 각 individual의 total distance로 정렬
        population_ranked = rank_individuals(cities, population)
        #population_ranked = rank_populations(cities, population)

        if i % 10 == 0:
            print(f"Generation {i + 1}")
            print("Best: ", population_ranked[0][0])
            print("Worst: ", population_ranked[-1][0])

        # 가장 성능이 좋은 individual을 elite_size만큼 선택
        elite_individuals = selection(population_ranked, ELITE_SIZE)
        
        # 교배를 통한 자식 세대 생성
        offspring = breed_population(population, elite_individuals, elite_size, mutation_rate)
        population = offspring

        # 이번 세대의 가장 성능이 좋은 individual
        current_best_distance, current_best_individual = population_ranked[0]

        # 이번 세대가 전 세대들보다 성능이 개선되었을 경우 best_distance, best_individual 갱신
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_individual = current_best_individual
            print(f"Generation {i + 1}: Best distance: {best_distance}")

    return best_distance, extend_individual(best_individual)

def genetic_algorithm_without_cluster(cities: List[City], pop_size: int, elite_size: int, mutation_rate: float, generations: int) -> Tuple[float, List[City]]:
    # 초기 population 생성
    #population = create_initial_population(cities, clusters, pop_size)
    temp_cluster = []
    temp_cluster.append(cities)
    #population = create_initial_population(cities, temp_cluster, pop_size)
    population = create_population(cities, pop_size)

    # 초기 최고해, 최단 거리 설정
    best_individual = None
    best_distance = float('inf')

    print("유전 알고리즘 시작")
    for i in tqdm(range(generations)):
        # 100세대마다 결과 저장
        if i >0 and i%100 == 0:
            best_order = extend_individual(best_individual)
            os.mkdir(f"results/{EXP_NAME}/GEN{i}")
            plt.figure(figsize=(12, 8))
            for i, city in enumerate(best_order):
                plt.scatter(city.x, city.y, c='blue', edgecolors='k', s=2)
                next_city = best_order[i + 1] if i + 1 < len(best_order) else best_order[0]
                plt.plot([city.x, next_city.x], [city.y, next_city.y],c='skyblue',alpha=0.5)
            
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"Best TSP Route (Total Distance: {best_distance:.2f})")
            plt.savefig(f'results/{EXP_NAME}/GEN{i}/test_CL{NUM_CLUSTER}_POP{POP_SIZE}.png')

            with open(f'results/{EXP_NAME}/GEN{i}/test_CL{NUM_CLUSTER}_POP{POP_SIZE}.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for city in best_order:
                    writer.writerow([city.x, city.y])
            
        #population_ranked = rank_individuals(cities, population)
        population_ranked = rank_populations(cities, population)

        if i % 10 == 0:
            print(f"Generation {i + 1}")
            print("Best: ", population_ranked[0][0])
            print("Worst: ", population_ranked[-1][0])

        # 가장 성능이 좋은 individual을 elite_size만큼 선택
        elite_individuals = selection(population_ranked, ELITE_SIZE)
        
        # 교배를 통한 자식 세대 생성
        offspring = breed_population(population, elite_individuals, elite_size, mutation_rate)
        population = offspring
        
         # 이번 세대의 가장 성능이 좋은 individual
        current_best_distance, current_best_individual = population_ranked[0]
        print(f"Generation {i + 1} Best distance: {current_best_distance}")
        
        # 이번 세대가 전 세대들보다 성능이 개선되었을 경우 best_distance, best_individual 갱신
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_individual = current_best_individual
            print(f"Generation {i + 1}: Best distance: {best_distance}")

    temp = []
    temp.append(best_individual)
    return best_distance, extend_individual(temp)