import numpy as np

def compute_features(coords, features):

    np.set_printoptions(precision=3)
    
    n = len(coords)
    m = n//2
    
    coords = np.array(coords, dtype=np.int32).reshape((m, 2))
    
    diff = np.moveaxis(coords[..., np.newaxis] - coords.T, -1, 1)       
    norm_m = np.einsum('...k,...k', diff, diff)
    
    features['d_sum'].append(np.linalg.norm(norm_m))
    
    cov_m = np.exp(-np.power(norm_m, 0.5)/20.0)    
    w, _ = np.linalg.eigh(cov_m)
    
    for i in range(0, w.shape[0]):     
        features[f'eig_{i}'].append(w[i]) 
    
    dx, dy = np.amax(coords, axis=0)-np.amin(coords, axis=0)

    features['long_side'].append(float(max(dx, dy)))    
    features['short_side'].append(float(min(dx, dy)))
           
    features['bias'].append(float(np.linalg.norm(np.mean(coords, axis=0))))
    
    # . . . . . x
    # . . . . x x
    # . . . x x x
    # . . . x x x
    # . . . x x x
    # . . . x x x
    _range = [0,1,2,3,4,5]
    dict_nearby = {}
    
    for i in _range:
        for j in _range:
            if(i <=j and (i >= 3 or j >= 3)):
                dict_nearby[(i, j)] = 0
                
    for i in range(0, m-1):
        for j in range(i+1, m):
            v = np.abs(coords[i]-coords[j])
            
            if(v[0] > v[1]):
                v = v[::-1]           
            if (v[0], v[1]) in dict_nearby.keys():
                #dict_nearby[(v[0], v[1])] += 1
                dict_nearby[(v[0], v[1])] = 1
                
    keys_list = list(dict_nearby.keys())
    
    for i in range(0, len(keys_list)):
        features[f'nearby_{i}'].append(dict_nearby[keys_list[i]])   
