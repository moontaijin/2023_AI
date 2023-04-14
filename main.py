from TreeSearch import *
from GA import *

def main():
    if not os.path.exists(f"./results/{EXP_NAME}"):
        os.mkdir(f"./results/{EXP_NAME}")
    # K-means 클러스터링으로 도시들을 클러스터로 분할
    clusters = cluster_cities(CITY_LIST, NUM_CLUSTER)

    # 각 클러스터에 대해 최적해를 찾음 (트리 탐색 사용)
    subproblem_solutions = solve_subproblems(clusters)
    # subproblem_solutions = []
    #best_distance = 0
    # for cluster in clusters:
    #     subproblem_solutions += [christofides_algorithm(cluster)]
        #best_distance += compute_total_distance(subproblem_solutions[-1])

    #best_order = extend_individual(subproblem_solutions) 
    
    #subproblem_solutions = solve_approximate_problems(CITY_LIST)
    best_distance, best_order = genetic_algorithm(CITY_LIST, subproblem_solutions, POP_SIZE, ELITE_SIZE, MUTATION_RATE, GENERATIONS)
    #print(f"Best distance: {best_distance}")
    #print(f"Best order: {best_order}")

    # 경로 시각화
    plt.figure(figsize=(12, 8))
    for i, city in enumerate(best_order):
        plt.scatter(city.x, city.y, c='blue', edgecolors='k', s=2)
        #plt.text(city.x - 5, city.y - 5, f"{i}", fontsize=12)
        next_city = best_order[i + 1] if i + 1 < len(best_order) else best_order[0]
        plt.plot([city.x, next_city.x], [city.y, next_city.y],c='skyblue',alpha=0.5)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Best TSP Route (Total Distance: {best_distance:.2f})")
    plt.savefig(f'results/{EXP_NAME}Best.png')

    # 결과 저장
    with open(f'results/{EXP_NAME}Best.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for city in best_order:
            writer.writerow([city.x, city.y])

if __name__ == '__main__':
    main()  