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
    visited_cities = [cities[start_city]]  # 시작 도시를 방문 목록에 추가하고 목록에서 제거
    cities.remove(cities[start_city])
    while cities:
        # 현재 도시에서 가장 가까운 도시를 찾음
        closest_city = min(cities, key=lambda city: visited_cities[-1].distance(city))
        visited_cities.append(closest_city)  # 가장 가까운 도시를 방문 목록에 추가
        cities.remove(closest_city)  # 가장 가까운 도시를 목록에서 제거

    return visited_cities

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
        distance, path = a_star(cluster,cluster[0])
        solutions.append([cluster[i] for i in path])
    return solutions

def solve_approximate_problems(cities):
    print("근사 해 찾기")
    path = greedy(cities,0)
    return path