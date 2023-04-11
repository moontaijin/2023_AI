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

def solve_subproblems(clusters):
    solutions = []
    print("클러스터 최적해 찾기")
    for cluster in tqdm(clusters):
        #distance, path = dfs(cluster, cluster[0], [0], 0)
        #distance, path = a_star(cluster,cluster[0])
        path = greedy(cluster,cluster[0])
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
    improve = 0
    max_try = 10000
    while improve < max_try:
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
                    improve = 0
                    best_path = deepcopy(new_path)
                    current_path = deepcopy(new_path)
                else:
                    improve += 1
                if improve >= max_try:
                    break
            if improve >= max_try:
                break
    
    # 최적 경로 반환
    best_route = [cities[best_path[i]] for i in range(len(best_path))]
    return best_route