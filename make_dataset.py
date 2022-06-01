import numpy as np
import parse as p
import pandas as pd
import argparse

from collections import defaultdict
from features import compute_features

def fetch_data(file_name):
    f = open(file_name, 'r')

    h_str = f.readline()   
    
    h_p = p.parse('Grid: {}, items: {}, count: {}', h_str)
    grid, items, count = int(h_p[0]), int(h_p[1]), int(h_p[2])
    
    data_info = {'grid': grid, 'items': items, 'count': count}    
    data_in_list = []
    
    pos = 0
    choice = random_choice(lim, dataset_half, count)
    
    for i in range(0, count):
        score_str = f.readline()
        data_str = f.readline()
            
        if i == choice[pos]:
            if pos < len(choice)-1:
                pos += 1
        else:
            continue
            
        print(i, len(data_in_list))
        
        score_p = p.parse("Score: {}", score_str)   
        score = int(score_p[0])
   
        coords_p_list = p.findall("{:d}", data_str)
        coords = [int(p_coords[0]) for p_coords in coords_p_list]
        
        data_in_list.append((coords, score))
        
    return data_in_list, data_info

def random_choice(lim, dataset_half, count):
    np.random.seed(123)    
    high = np.random.choice(lim, dataset_half, replace=False)
    low = np.random.choice(count - lim, dataset_half, replace=False) + lim

    return np.sort(np.concatenate((high, low)))
   
if __name__ == "__main__": 
    cmd_parser = argparse.ArgumentParser()

    cmd_parser.add_argument('--lim', type=int, default=25000)
    cmd_parser.add_argument('--dataset_half', type=int, default=5000)
    
    cmd_args = cmd_parser.parse_args()           
    lim, dataset_half = cmd_args.lim, cmd_args.dataset_half
    print(lim, dataset_half)
        
    data, data_info = fetch_data('dfs_merged.txt')

    features = defaultdict(list)
    
    for i in range(0, len(data)):
        feat_vec = compute_features(data[i][0], features)        
        features['score'].append(data[i][1])
        
    df = pd.DataFrame(data=features)
    df.to_csv('stepping_stones_puzzle_dataset.csv')    
