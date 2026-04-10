# ev-station-ga

A genetic algorithm (GA) that optimizes the placement of EV charging stations across a simulated 20x20 mile city grid. The GA balances two competing objectives — maximizing demand coverage and minimizing installation cost — using a weighted fitness function.

## Features
- Tournament selection, uniform crossover, and swap mutation
- Elitism to preserve the best solution across generations
- Chromosome repair to always maintain a valid number of stations
- Four experiments comparing baseline, high mutation, large population, and cost-biased configurations
- Convergence plots and city map visualization saved as `.png` files

## Platform
Windows, macOS, or Linux. Requires Python 3.10+.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```
Outputs convergence plots and a city map to `convergence.png` and `city_map.png`.

## Note
The city grid, candidate station locations, and demand points are synthetically generated for experimentation purposes and do not represent a real city.