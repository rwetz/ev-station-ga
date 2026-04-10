import random
import math
import matplotlib.pyplot as plt
import numpy as np

#config
RANDOM_SEED       = 42          # for reproducibility
GRID_SIZE         = 20.0        # city is 20x20 miles
N_CANDIDATES      = 10          # candidate charging station locations
N_DEMAND_POINTS   = 50          # resident / demand locations
K_STATIONS        = 3           # number of stations to place
COVERAGE_RADIUS   = 5.0         # miles — a station covers demand within this radius

#GA parameters (these are changed in experiments)
POP_SIZE          = 50
N_GENERATIONS     = 200
CROSSOVER_RATE    = 0.85
MUTATION_RATE     = 0.15        # probability of applying swap mutation per chromosome
TOURNAMENT_K      = 4           # tournament size for selection
W_COVERAGE        = 0.7         # weight for coverage in fitness
W_COST            = 0.3         # weight for cost in fitness

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#dataset

def generate_dataset():
    #manually seeded to create an interesting optimization landscape
    candidate_coords = [
        (4.0,  4.0),   
        (4.0,  16.0),  
        (16.0, 4.0),   
        (16.0, 16.0),  
        (10.0, 10.0),  
        (7.0,  10.0),  
        (13.0, 10.0),  
        (10.0, 6.0),   
        (10.0, 14.0),  
        (6.0,  7.0),   
    ]

    #installation costs 
    costs = [80, 75, 70, 72, 200, 150, 145, 130, 135, 110]

    #demand points
    np.random.seed(RANDOM_SEED)
    demand_coords = []
    for _ in range(25):
        x = np.random.normal(10.0, 2.5)
        y = np.random.normal(10.0, 2.5)
        demand_coords.append((float(np.clip(x, 0, GRID_SIZE)),
                               float(np.clip(y, 0, GRID_SIZE))))
    for _ in range(25):
        x = np.random.uniform(0, GRID_SIZE)
        y = np.random.uniform(0, GRID_SIZE)
        demand_coords.append((x, y))

    return candidate_coords, costs, demand_coords


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def build_coverage_matrix(candidate_coords, demand_coords, radius):
    n = len(candidate_coords)
    m = len(demand_coords)
    coverage = [[0] * m for _ in range(n)]
    for i, c in enumerate(candidate_coords):
        for j, d in enumerate(demand_coords):
            if euclidean_distance(c, d) <= radius:
                coverage[i][j] = 1
    return coverage

#fitness func as described in the report
def fitness(chromosome, coverage_matrix, costs, k,
            w_cov=W_COVERAGE, w_cost=W_COST):
    selected = [i for i, gene in enumerate(chromosome) if gene == 1]
    n_selected = len(selected)

    #penalty
    penalty = 0.0
    if n_selected != k:
        penalty = 1.0 * abs(n_selected - k)

    if n_selected == 0:
        return -penalty

    #coverage
    n_demand = len(coverage_matrix[0])
    covered = set()
    for i in selected:
        for j in range(n_demand):
            if coverage_matrix[i][j] == 1:
                covered.add(j)
    coverage_ratio = len(covered) / n_demand

    #cost ratio 
    sorted_costs = sorted(costs, reverse=True)
    max_possible_cost = sum(sorted_costs[:k])
    total_cost = sum(costs[i] for i in selected)
    cost_ratio = total_cost / max_possible_cost

    score = w_cov * coverage_ratio - w_cost * cost_ratio - penalty
    return score


def decode_solution(chromosome, coverage_matrix, costs, k):
    selected = [i for i, gene in enumerate(chromosome) if gene == 1]
    n_demand = len(coverage_matrix[0])
    covered = set()
    for i in selected:
        for j in range(n_demand):
            if coverage_matrix[i][j] == 1:
                covered.add(j)
    coverage_pct = len(covered) / n_demand * 100
    total_cost = sum(costs[i] for i in selected)
    return selected, coverage_pct, total_cost, len(covered)

#ga operations
def create_chromosome(n, k):
    chrom = [0] * n
    selected = random.sample(range(n), k)
    for i in selected:
        chrom[i] = 1
    return chrom


def initialize_population(pop_size, n, k):
    return [create_chromosome(n, k) for _ in range(pop_size)]


def repair(chromosome, k):
    chrom = chromosome[:]
    ones  = [i for i, g in enumerate(chrom) if g == 1]
    zeros = [i for i, g in enumerate(chrom) if g == 0]

    while len(ones) > k:
        remove = random.choice(ones)
        chrom[remove] = 0
        ones.remove(remove)
        zeros.append(remove)

    while len(ones) < k:
        add = random.choice(zeros)
        chrom[add] = 1
        zeros.remove(add)
        ones.append(add)

    return chrom


def tournament_select(population, fitnesses, tournament_k):
    contestants = random.sample(range(len(population)), tournament_k)
    best = max(contestants, key=lambda idx: fitnesses[idx])
    return population[best][:]


def uniform_crossover(parent1, parent2, crossover_rate):
    if random.random() > crossover_rate:
        return parent1[:], parent2[:]

    n = len(parent1)
    child1, child2 = [], []
    for i in range(n):
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2


def swap_mutate(chromosome, mutation_rate, k):
    if random.random() > mutation_rate:
        return chromosome

    chrom = chromosome[:]
    ones  = [i for i, g in enumerate(chrom) if g == 1]
    zeros = [i for i, g in enumerate(chrom) if g == 0]

    if ones and zeros:
        turn_off = random.choice(ones)
        turn_on  = random.choice(zeros)
        chrom[turn_off] = 0
        chrom[turn_on]  = 1

    return chrom

#main loop
def run_ga(coverage_matrix, costs, n, k,
           pop_size=POP_SIZE, n_generations=N_GENERATIONS,
           crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
           tournament_k=TOURNAMENT_K, w_cov=W_COVERAGE, w_cost=W_COST,
           verbose=True):
    population = initialize_population(pop_size, n, k)
    best_chromosome = None
    best_fitness = float('-inf')
    best_fitness_history = []
    avg_fitness_history  = []

    for gen in range(n_generations):
        #evaluate fitness
        fitnesses = [fitness(chrom, coverage_matrix, costs, k, w_cov, w_cost)
                     for chrom in population]

        #track best
        gen_best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
        gen_best_fit = fitnesses[gen_best_idx]
        gen_avg_fit  = sum(fitnesses) / pop_size

        if gen_best_fit > best_fitness:
            best_fitness   = gen_best_fit
            best_chromosome = population[gen_best_idx][:]

        best_fitness_history.append(gen_best_fit)
        avg_fitness_history.append(gen_avg_fit)

        if verbose and (gen % 50 == 0 or gen == n_generations - 1):
            print(f"  Gen {gen:>4}: Best Fitness = {gen_best_fit:.4f} | "
                  f"Avg = {gen_avg_fit:.4f}")

        #elitism implementation
        elite = population[gen_best_idx][:]

        #build next generation
        new_population = [elite]

        while len(new_population) < pop_size:
            parent1 = tournament_select(population, fitnesses, tournament_k)
            parent2 = tournament_select(population, fitnesses, tournament_k)

            child1, child2 = uniform_crossover(parent1, parent2, crossover_rate)

            #repair children to have exactly k stations
            child1 = repair(child1, k)
            child2 = repair(child2, k)

            #mutation
            child1 = swap_mutate(child1, mutation_rate, k)
            child2 = swap_mutate(child2, mutation_rate, k)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    return best_chromosome, best_fitness_history, avg_fitness_history

#graphing data 
def plot_convergence(histories, labels, title="GA Convergence"):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
    for (best_hist, avg_hist), label, color in zip(histories, labels, colors):
        ax.plot(best_hist, label=f"{label} — Best", color=color, linewidth=2)
        ax.plot(avg_hist,  label=f"{label} — Avg",  color=color, linewidth=1,
                linestyle='--', alpha=0.6)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Score")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150)
    plt.show()
    print("Saved: convergence.png")


def plot_city(candidate_coords, demand_coords, selected_stations, costs,
              coverage_radius=COVERAGE_RADIUS):
    fig, ax = plt.subplots(figsize=(8, 8))

    #demand points
    dx = [p[0] for p in demand_coords]
    dy = [p[1] for p in demand_coords]
    ax.scatter(dx, dy, c='#90CAF9', s=30, zorder=2, label='Demand Points')

    #all candidates (not selected)
    not_selected = [i for i in range(len(candidate_coords))
                    if i not in selected_stations]
    for i in not_selected:
        x, y = candidate_coords[i]
        ax.scatter(x, y, c='gray', s=80, marker='s', zorder=3)
        ax.annotate(f"C{i}\n${costs[i]}K", (x, y),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, color='gray')

    #selected stations + coverage circles
    colors_sel = ['#E53935', '#43A047', '#8E24AA']
    for idx, i in enumerate(selected_stations):
        x, y = candidate_coords[i]
        color = colors_sel[idx % len(colors_sel)]
        ax.scatter(x, y, c=color, s=200, marker='*', zorder=5,
                   label=f"Station {i} (${costs[i]}K)")
        circle = plt.Circle((x, y), coverage_radius, color=color, alpha=0.12, zorder=1)
        ax.add_patch(circle)
        ax.annotate(f"C{i}", (x, y),
                    textcoords="offset points", xytext=(5, 5), fontsize=9,
                    fontweight='bold', color=color)

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xlabel("Miles (East-West)")
    ax.set_ylabel("Miles (North-South)")
    ax.set_title(f"Optimal EV Charging Station Placement\n"
                 f"(Coverage radius = {coverage_radius} miles)")
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("city_map.png", dpi=150)
    plt.show()
    print("Saved: city_map.png")

#testing
def run_experiment(label, candidate_coords, demand_coords, costs, coverage_matrix,
                   **ga_kwargs):
    n = len(candidate_coords)
    k = K_STATIONS
    print(f"\n{'='*55}")
    print(f"EXPERIMENT: {label}")
    print(f"{'='*55}")
    best_chrom, best_hist, avg_hist = run_ga(
        coverage_matrix, costs, n, k, verbose=True, **ga_kwargs
    )
    selected, cov_pct, total_cost, n_covered = decode_solution(
        best_chrom, coverage_matrix, costs, k
    )
    best_fitness_val = fitness(best_chrom, coverage_matrix, costs, k)
    print(f"\n  Selected Stations : {selected}")
    print(f"  Coverage          : {cov_pct:.1f}% ({n_covered}/{N_DEMAND_POINTS} demand points)")
    print(f"  Total Cost        : ${total_cost}K")
    print(f"  Best Fitness Score: {best_fitness_val:.4f}")
    return best_chrom, best_hist, avg_hist, selected, cov_pct, total_cost

#main
if __name__ == "__main__":
    print("=" * 55)
    print("EV Charging Station Optimization — Genetic Algorithm")
    print("=" * 55)

    #dataset
    candidate_coords, costs, demand_coords = generate_dataset()
    coverage_matrix = build_coverage_matrix(
        candidate_coords, demand_coords, COVERAGE_RADIUS
    )
    n = len(candidate_coords)

    print(f"\nDataset Summary:")
    print(f"  Candidate locations : {n}")
    print(f"  Demand points       : {N_DEMAND_POINTS}")
    print(f"  Stations to place   : {K_STATIONS}")
    print(f"  Coverage radius     : {COVERAGE_RADIUS} miles")
    print(f"\nCandidate Costs ($K): {costs}")

    print(f"\nPer-candidate max coverage:")
    for i, (coord, cost) in enumerate(zip(candidate_coords, costs)):
        covered_count = sum(coverage_matrix[i])
        print(f"  Candidate {i} at {coord}: covers {covered_count}/{N_DEMAND_POINTS} pts, "
              f"cost=${cost}K")

    #experiments
    all_histories = []
    all_labels    = []

    #experiment 1: baseline
    chrom1, best1, avg1, sel1, cov1, cost1 = run_experiment(
        "Baseline (pop=50, mut=0.15, cx=0.85)",
        candidate_coords, demand_coords, costs, coverage_matrix,
        pop_size=50, n_generations=200, mutation_rate=0.15, crossover_rate=0.85
    )
    all_histories.append((best1, avg1))
    all_labels.append("Baseline")

    #experiment 2: high mutation rate
    chrom2, best2, avg2, sel2, cov2, cost2 = run_experiment(
        "High Mutation (mut=0.40)",
        candidate_coords, demand_coords, costs, coverage_matrix,
        pop_size=50, n_generations=200, mutation_rate=0.40, crossover_rate=0.85
    )
    all_histories.append((best2, avg2))
    all_labels.append("High Mutation")

    #experiment 3: large population
    chrom3, best3, avg3, sel3, cov3, cost3 = run_experiment(
        "Large Population (pop=150)",
        candidate_coords, demand_coords, costs, coverage_matrix,
        pop_size=150, n_generations=200, mutation_rate=0.15, crossover_rate=0.85
    )
    all_histories.append((best3, avg3))
    all_labels.append("Large Population")

    #experiment 4: cost-biased weights
    chrom4, best4, avg4, sel4, cov4, cost4 = run_experiment(
        "Cost-Biased (w_cov=0.4, w_cost=0.6)",
        candidate_coords, demand_coords, costs, coverage_matrix,
        pop_size=50, n_generations=200, mutation_rate=0.15, crossover_rate=0.85,
        w_cov=0.4, w_cost=0.6
    )
    all_histories.append((best4, avg4))
    all_labels.append("Cost-Biased")

    #convergence Plot
    plot_convergence(all_histories, all_labels,
                     title="GA Convergence — EV Charging Station Optimization")

    plot_city(candidate_coords, demand_coords, sel1, costs, COVERAGE_RADIUS)

    print("\n" + "="*55)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*55)
    print(f"{'Config':<30} {'Stations':<15} {'Coverage':>10} {'Cost ($K)':>12}")
    print("-"*70)
    results = [
        ("Baseline",          sel1, cov1, cost1),
        ("High Mutation",     sel2, cov2, cost2),
        ("Large Population",  sel3, cov3, cost3),
        ("Cost-Biased",       sel4, cov4, cost4),
    ]
    for name, sel, cov, cost in results:
        print(f"  {name:<28} {str(sel):<15} {cov:>9.1f}%  ${cost:>8}K")
    print("="*55)