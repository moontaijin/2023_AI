import numpy as np
import random
import pandas as pd
import heapq
import json
import csv
import networkx as nx

from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple
from math import sqrt
from tqdm import tqdm
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from copy import deepcopy

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