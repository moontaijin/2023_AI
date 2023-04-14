from base import *
from GA import *

# num_clusters 만큼 K-Means 클러스터 생성
def cluster_cities(cities, num_clusters):
    # 도시 좌표를 2차원 배열로 나타냄.
    city_coordinates = np.array([[city.x, city.y] for city in cities])
    # K-Means 알고리즘을 사용하여 도시를 클러스터링
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(city_coordinates)
    # 클러스터를 저장하기 위한 딕셔너리 생성
    clusters = {i: [] for i in range(num_clusters)}
    for city, cluster_label in zip(cities, kmeans.labels_):
        # 딕셔너리에 각각의 클러스터를 추가
        clusters[cluster_label].append(city)
    # 클러스터 리스트를 반환
    return list(clusters.values())

# 휴리스틱 function
def heuristic(city1, city2):
    # 2개의 도시 사이의 거리 반환
    return city1.distance(city2)

def greedy(cities, start_city):
    # 각 도시의 인덱스를 나타내는 리스트 생성
    city_indices = list(range(len(cities)))
    # 시작 도시의 인덱스 찾음
    start_city_index = cities.index(start_city)
    visited = [city_indices.pop(start_city_index)]  # 시작 도시 인덱스를 방문 목록에 추가하고 목록에서 제거
    while city_indices:
        # 현재 도시에서 가장 가까운 도시 인덱스를 찾음
        closest_city_index = min(city_indices, key=lambda index: cities[visited[-1]].distance(cities[index]))
        visited.append(closest_city_index)  # 가장 가까운 도시 인덱스를 방문 목록에 추가
        city_indices.remove(closest_city_index)  # 가장 가까운 도시 인덱스를 목록에서 제거

    # 모든 도시를 방분한 후, 방문한 도시 인덱스 목록을 반환
    return visited
def dfs(cities, current_city, visited, total_distance):
    # 모든 도시를 방문한 경우
    if len(visited) == len(cities):
        # 현재 도시에서 시작 도시로 돌아가는 거리와 빈 경로를 반환
        return total_distance + current_city.distance(cities[visited[0]]), []

    best_distance = float('inf')
    best_path = []
    for i, city in enumerate(cities):
        # 아직 방문하지 않은 도시인 경우
        if i not in visited:
            # 새로운 방문 목록을 만들고, 방문목록에 추가
            new_visited = visited.copy()
            new_visited.append(i)
            # 현재 도시에서 새로운 도시로의 방문거리와 경로 구함.
            new_distance, new_path = dfs(cities, city, new_visited, total_distance + current_city.distance(city))
            # 새로운 경로가 현재까지의 최단 경로보다 더 짧은 경우
            if new_distance < best_distance:
                best_distance = new_distance
                # 현재 도시를 경로의 첫번째로 추가
                best_path = [i] + best_path
    # 최단 거리와 경로를 반환.
    return best_distance, best_path

def a_star(cities, start_city):
    # 힙으로 구현된 우선순위 큐에 (우선순위, 현재 경로, 현재 거리, 방문한 도시 리스트)를 튜플 형태로 추가
    frontier = [(0, [start_city], 0, [0])]
    # heapify() 함수를 통해 우선순위 큐를 요소를 추가 / 삭제 할 수 있도록 힙 자료구조로 변환
    heapq.heapify(frontier)

    # 우선순위 큐가 비어있지 않은 동안 반복
    while frontier:
        # 현재 경로의 가장 마지막 도시를 pop 하면서, 현재 경로, 현재 거리, 방문한 도시 리스트를 변수로 할당
        _, current_path, current_distance, visited = heapq.heappop(frontier)

        # 방문한 도시의 수가 전체 도시의 수와 같으면
        if len(visited) == len(cities):
            #  마지막 도시에서 시작 도시로 돌아가는 거리와 방문한 도시 리스트를 반환
            return current_distance + current_path[-1].distance(cities[visited[0]]), visited

        # 전체 도시를 반복하면서, 아직 방문하지 않은 도시를 찾음
        for i, city in enumerate(cities):
            if i not in visited:
                # 새로 방문한 도시 리스트 생성
                new_visited = visited + [i]
                # 새로운 경로의 거리 계산
                new_distance = current_distance + current_path[-1].distance(city)
                # 새로운 경로의 우선순위 계산
                priority = new_distance + heuristic(current_path[-1], city)
                # (우선순위, 경로, 새 경로의 거리, 새로 방문한 도시 리스트) 튜플과 새로운 경로를 힙에 추가
                heapq.heappush(frontier, (priority, current_path + [city], new_distance, new_visited))

    # 도시를 모두 방문할 수 없는 경우, None 반환
    return None, None

def greedy_a_star(cities, start_city):
    # 각 도시의 인덱스를 리스트로 생성
    city_indices = list(range(len(cities)))
    start_city_index = cities.index(start_city)
    visited = [city_indices.pop(0)]  # 시작 도시 인덱스를 방문 목록에 추가하고 목록에서 제거
    while len(city_indices) >= 10:
        # 현재 도시에서 가장 가까운 도시 인덱스를 찾음
        closest_city_index = min(city_indices, key=lambda index: cities[visited[-1]].distance(cities[index]))
        visited.append(closest_city_index)  # 가장 가까운 도시 인덱스를 방문 목록에 추가
        city_indices.remove(closest_city_index)  # 가장 가까운 도시 인덱스를 목록에서 제거

    # 우선순위 큐(Heap)를 초기화하고, 현재 경로, 현재 까지의 거리, 방문한 도시의 리스트를 담은 튜플을 루트 노드로 추가
    frontier = [(0, [cities[visited[-1]]], 0, visited)]
    heapq.heapify(frontier)
    
    # 우선 순위 큐에서 노드를 하나씩 꺼내면서 탐색
    while frontier:
        # 노드에서 경로, 거리, 방문한 도시의 리스트를 꺼내줌
        _, current_path, current_distance, visited = heapq.heappop(frontier)

        # 모든 도시를 방문했다면, 방문한 도시의 리스트를 반환
        if len(visited) == len(cities):
            return visited

        # 모든 도시를 방문하지 않았다면, 현재 도시에서 갈 수 있는 모든 도시를 탐색하여 우선순위 큐에 추가
        for i, city in enumerate(cities):
            # 방문하지 않은 도시에 대해서만 처리
            if i not in visited:
                # 새로운 도시를 방문한 경우의 경로, 거리, 방문한 도시의 리스트를 계산
                new_visited = visited + [i]
                new_distance = current_distance + current_path[-1].distance(city)
                priority = new_distance + heuristic(current_path[-1], city)
                # 계산한 경로, 거리, 방문한 도시의 리스트를 담은 튜플을 우선순위 큐에 추가
                heapq.heappush(frontier, (priority, current_path + [city], new_distance, new_visited))

    # 모든 도시를 방문하지 못한 경우 None 반환
    return None

def christofides_algorithm(points: List[City]):
    # 입력받은 도시 수를 저장
    num_points = len(points)

    # 도시들의 좌표 값을 numpy array 형태로 변환
    points = [np.array([city.x,city.y]) for city in points]

    # 도시들 간의 거리를 계산하여 거리 matrix 생성
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

    # 변환된 경로를 city 클래스 객체로 변환하여 반환
    tsp_path = [City(city[0],city[1]) for city in tsp_path]
    return tsp_path

def solve_subproblems(clusters):
    # 각 클러스터에 대한 최적해를 저장할 리스트
    solutions = []

    print("클러스터 최적해 찾기")

    # tqdm 라이브러리를 이용하여 각 클러스터 마다 순회
    for cluster in tqdm(clusters):
        # 현재 클러스터에서 greedy_a_star를 이용하여 시작 도시에서부터 최적 경로를 구함
        path = greedy_a_star(cluster, cluster[0])

        # 2-opt 알고리즘을 이용하여 구한 경로를 최적화
        solutions.append(two_opt([cluster[i] for i in path]))

    # 모든 클러스터에 대해 최적해를 담은 리스트를 반환
    return solutions

def two_opt(cities):
    # 시작 경로 설정
    current_path = [i for i in range(len(cities))]
    best_path = deepcopy(current_path)
    
    # 경로 최적화
    improve = True
    while improve:
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
    
    # 최적 경로 반환
    best_route = [cities[best_path[i]] for i in range(len(best_path))]
    return best_route
