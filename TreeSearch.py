from base import *
from GA import *

def cluster_cities(cities, num_clusters):
    city_coordinates = np.array([[city.x, city.y] for city in cities])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(city_coordinates)
    clusters = {i: [] for i in range(num_clusters)}
    for city, cluster_label in zip(cities, kmeans.labels_):
        clusters[cluster_label].append(city)
    return list(clusters.values())

def heuristic(city1, city2):
    return city1.distance(city2)

def greedy(cities, start_city):
    # city_indices = list(range(len(cities)))
    # start_city_index = cities.index(start_city)
    # visited = [city_indices.pop(0)]  # 시작 도시 인덱스를 방문 목록에 추가하고 목록에서 제거
    # while city_indices:
    #     # 현재 도시에서 가장 가까운 도시 인덱스를 찾음
    #     closest_city_index = min(city_indices, key=lambda index: cities[visited[-1]].distance(cities[index]))
    #     visited.append(closest_city_index)  # 가장 가까운 도시 인덱스를 방문 목록에 추가
    #     city_indices.remove(closest_city_index)  # 가장 가까운 도시 인덱스를 목록에서 제거

    # return visited

    if len(cities) == 1:
        return [0]
    else:
        center = City(50, 50)
        city_indices = list(range(len(cities)))
        first = min(city_indices, key=lambda index: cities[index].distance(center))
        visited = [first] 
        city_indices.remove(first)

        second = min(city_indices, key=lambda index: cities[index].distance(center))
        city_indices.remove(second)

        while city_indices:
            # 현재 도시에서 가장 가까운 도시 인덱스를 찾음
            closest_city_index = min(city_indices, key=lambda index: cities[visited[-1]].distance(cities[index]))
            visited.append(closest_city_index)  # 가장 가까운 도시 인덱스를 방문 목록에 추가
            city_indices.remove(closest_city_index)  # 가장 가까운 도시 인덱스를 목록에서 제거
        
        visited.append(second)
        print(visited)
        return visited

def dfs(cities, current_city, visited, total_distance):
    if len(visited) == len(cities):
        return total_distance + current_city.distance(cities[visited[0]]), []

    best_distance = float('inf')
    best_path = []
    for i, city in enumerate(cities):
        if i not in visited:
            new_visited = visited.copy()
            new_visited.append(i)
            new_distance, new_path = dfs(cities, city, new_visited, total_distance + current_city.distance(city))
            if new_distance < best_distance:
                best_distance = new_distance
                best_path = [i] + best_path

    return best_distance, best_path

def a_star(cities, start_city):
    frontier = [(0, [start_city], 0, [0])]
    heapq.heapify(frontier)

    while frontier:
        _, current_path, current_distance, visited = heapq.heappop(frontier)

        if len(visited) == len(cities):
            return current_distance + current_path[-1].distance(cities[visited[0]]), visited
        
        for i, city in enumerate(cities):
            if i not in visited:
                new_visited = visited + [i]
                new_distance = current_distance + current_path[-1].distance(city)
                priority = new_distance + heuristic(current_path[-1], city)
                heapq.heappush(frontier, (priority, current_path + [city], new_distance, new_visited))

    return None, None

def greedy_a_star(cities, start_city):
    city_indices = list(range(len(cities)))
    start_city_index = cities.index(start_city)
    visited = [city_indices.pop(0)]  # 시작 도시 인덱스를 방문 목록에 추가하고 목록에서 제거
    while len(city_indices) >= 10:
        # 현재 도시에서 가장 가까운 도시 인덱스를 찾음
        closest_city_index = min(city_indices, key=lambda index: cities[visited[-1]].distance(cities[index]))
        visited.append(closest_city_index)  # 가장 가까운 도시 인덱스를 방문 목록에 추가
        city_indices.remove(closest_city_index)  # 가장 가까운 도시 인덱스를 목록에서 제거
    
    frontier = [(0, [cities[visited[-1]]], 0, visited)]
    heapq.heapify(frontier)

    while frontier:
        _, current_path, current_distance, visited = heapq.heappop(frontier)

        if len(visited) == len(cities):
            return visited
        
        for i, city in enumerate(cities):
            if i not in visited:
                new_visited = visited + [i]
                new_distance = current_distance + current_path[-1].distance(city)
                priority = new_distance + heuristic(current_path[-1], city)
                heapq.heappush(frontier, (priority, current_path + [city], new_distance, new_visited))

    return None

def christofides_algorithm(points: List[City]):
    num_points = len(points)
    points = [np.array([city.x,city.y]) for city in points]
    dist_matrix = squareform(pdist(points))

    # 최소 신장 트리 찾기
    G = nx.from_numpy_array(dist_matrix)
    MST = nx.minimum_spanning_tree(G)

    # 홀수 차수 정점 찾기
    odd_degree_nodes = [v for v, d in MST.degree() if d % 2 == 1]

    # 최소 완전 매칭 찾기
    odd_deg_graph = nx.subgraph(G, odd_degree_nodes)
    MWPM = nx.max_weight_matching(odd_deg_graph, maxcardinality=True, weight='weight')

    # 오일러 회로 형성을 위해 그래프에 최소 완전 매칭 추가
    MST.add_edges_from(MWPM)
    eulerian_circuit = nx.eulerian_circuit(MST)

    # 해밀턴 회로로 변환
    visited = set()
    tsp_path = []
    for _, v in eulerian_circuit:
        if v not in visited:
            tsp_path.append(points[v])
            visited.add(v)

    tsp_path = [City(city[0],city[1]) for city in tsp_path]
    return tsp_path

def solve_subproblems(clusters):
    solutions = []
    print("클러스터 최적해 찾기")
    for cluster in tqdm(clusters):
        #distance, path = dfs(cluster, cluster[0], [0], 0)
        #distance, path = a_star(cluster,cluster[0])

        #sub_clusters = cluster_cities(cluster, 5)
        path = greedy(cluster, cluster[0])

        #best_cluster_distance, best_path_cluster = genetic_algorithm_without_cluster(cluster, POP_SIZE, ELITE_SIZE, MUTATION_RATE, 500)
        #best_cluster_distance, best_path_cluster = genetic_algorithm(cluster, sub_clusters, POP_SIZE, ELITE_SIZE, MUTATION_RATE, 1000)

        #path = greedy(cluster,cluster[0])
        #path = greedy(best_path_cluster, best_path_cluster[0])
        #print(path)
        # 
        #path = greedy(best_path_cluster, best_path_cluster[0])
        solutions.append(two_opt([cluster[i] for i in path]))
    return solutions

def solve_approximate_problems(cities):
    print("근사 해 찾기")
    path = greedy(cities,cities[0])
    return path

def two_opt(cities):
    # 시작 경로 설정
    current_path = [i for i in range(len(cities))]
    best_path = deepcopy(current_path)
    
    # 경로 최적화
    improve = True
    max_try = 0
    while improve:
        print(max_try)
        max_try = max_try + 1
        improve = False
        for i in range(1, len(cities) - 1):
            for j in range(i + 1, len(cities)):
                # 경로 일부를 뒤집음
                new_path = deepcopy(current_path)
                new_path[i:j] = current_path[i:j][::-1]
                
                # 경로 길이 계산
                current_distance = 0
                new_distance = 0
                for k in range(len(current_path)):
                    current_distance += cities[current_path[k]].distance(cities[current_path[(k + 1) % len(current_path)]])
                    new_distance += cities[new_path[k]].distance(cities[new_path[(k + 1) % len(new_path)]])

                # 경로 업데이트
                if new_distance < current_distance:
                    improve = True
                    best_path = deepcopy(new_path)
                    current_path = deepcopy(new_path)
            #     else:
            #         improve += 1
            #     if improve >= max_try:
            #         break
            # if improve >= max_try:
            #     break
    
    # 최적 경로 반환
    best_route = [cities[best_path[i]] for i in range(len(best_path))]
    return best_route
