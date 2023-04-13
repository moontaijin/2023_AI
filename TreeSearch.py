from base import *

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
    city_indices = list(range(len(cities)))
    start_city_index = cities.index(start_city)
    visited = [city_indices.pop(0)]  # 시작 도시 인덱스를 방문 목록에 추가하고 목록에서 제거
    while city_indices:
        # 현재 도시에서 가장 가까운 도시 인덱스를 찾음
        closest_city_index = min(city_indices, key=lambda index: cities[visited[-1]].distance(cities[index]))
        visited.append(closest_city_index)  # 가장 가까운 도시 인덱스를 방문 목록에 추가
        city_indices.remove(closest_city_index)  # 가장 가까운 도시 인덱스를 목록에서 제거

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
        #path = greedy(cluster,cluster[0])
        path = greedy_a_star(cluster,cluster[0])
        solutions.append([cluster[i] for i in path])
    return solutions