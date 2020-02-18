import numpy as np
import pandas as pd

def load_data():
  a = pd.read_csv('adult.data', header=None,
                  names=['age', 'workclass', 'fnlwgt', 'education',
                        'education-num', 'marital-status', 'occupation', 'relationship',
                        'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country', 'income'])
  print(a.shape)
  b = pd.read_csv('adult.test', header=None,
                  names=['age', 'workclass', 'fnlwgt', 'education',
                        'education-num', 'marital-status', 'occupation', 'relationship',
                        'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country', 'income'])
  print(b.shape)
  full = pd.concat([a, b], axis=0)
  print(full.shape)

  full = full.drop('education', axis=1)
  full = full.drop('native-country', axis=1)
  full = full.drop('fnlwgt', axis=1)
  for col in ['workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'income']:
    if '-' in col:
      prefix_col = col.split('-')[0]
    else:
      prefix_col = col
    # kind of converting into one-hot encoding, basically coverting categorical values into one-hott encoding
    full = pd.concat([full, pd.get_dummies(full[col], prefix=prefix_col, drop_first=True)], axis=1)
    full = full.drop(col, axis=1)

  # normalize the numeric features
  cols_to_norm = ['capital-gain','capital-loss','hours-per-week']
  full[cols_to_norm] = full[cols_to_norm].apply(lambda x: (x - x.min()) / x.max())

  # print(full.shape)
  # print(list(full.columns))
  # print(full.tail())
  # print("statistics of data")
  # print(full.describe())

  # ‘age' is somewhat hard to normalize, so, will normalize it in the numpy array’
  full_np = full.to_numpy()
  y = (full_np[:, -1] + full_np[:, -2]).astype(np.float32)
  y = np.delete(y, 32561, axis=0) # this row is '1x3 Cross validator' and should be removed
  x = np.delete(full_np, [full_np.shape[1]-1, full_np.shape[1]-2, full_np.shape[1]-3], axis=1)
  x = np.delete(x, 32561, axis=0).astype(np.float32)


  x[:,0] = (x[:,0]-np.amin(x[:,0]))/(np.amax(x[:,0])-np.amin(x[:,0]))

  trn_x, trn_y = x[:32561], y[:32561]
  tst_x, tst_y = x[32561:], y[32561:]

  trn_zero_inds = np.where(trn_y==0)[0]
  trn_one_inds = np.where(trn_y==1)[0]
  tst_zero_inds = np.where(tst_y==0)[0]
  tst_one_inds = np.where(tst_y==1)[0]

  # discrete feature indocator, used for computing distance with mixture of l2,l1 distance 
  disc_idx = np.ones(trn_x.shape[1])
  disc_idx[0:4] = 0 # first four features are numerical values

  # subsampling to make the dataset balanced
  trn_zeros = np.random.choice(trn_zero_inds.shape[0], trn_one_inds.shape[0], replace=False)
  tst_zeros = np.random.choice(tst_zero_inds.shape[0], tst_one_inds.shape[0], replace=False)

  trn_x = np.concatenate((trn_x[trn_zeros], trn_x[trn_one_inds]), axis=0)
  tst_x = np.concatenate((tst_x[tst_zeros], tst_x[tst_one_inds]), axis=0)
  trn_y = np.concatenate((trn_y[trn_zeros], trn_y[trn_one_inds]), axis=0)
  tst_y = np.concatenate((tst_y[tst_zeros], tst_y[tst_one_inds]), axis=0)

  trn_shuffle = np.random.choice(trn_x.shape[0], trn_x.shape[0], replace=False)
  trn_x, trn_y = trn_x[trn_shuffle], trn_y[trn_shuffle]
  return trn_x, trn_y, tst_x, tst_y, full

  # define distance computaton related functions
def compute_dist(src, dest,disc_idx= None):
  # compute the distance of two points, if there are discrete indices: compute their hamming distance
  if disc_idx is not None:
    dist = np.linalg.norm(src[np.logical_not(disc_idx)]-dest[np.logical_not(disc_idx)]) + np.linalg.norm(src[disc_idx] - dest[disc_idx],ord = 1)
    return dist
  else:
    return np.linalg.norm(src-dest)

def compactness(subpop):
  # find the compactness of the given subpopulation
  # calculate the variance of the data points
  mean = np.mean(subpop,axis = 0)
  dist = 0
  for i in range(len(subpop)):
    dist += compute_dist(subpop[i], mean)**2 
  return dist/len(subpop)

def separability(subpop,restpops, metric = 'min'):
  # find the separability of given points from the 
  dist = []
  for restpop in restpops:
    if metric == 'min':# max distance of two points, each in the cluster
      min_dist = 1e10
      for i in range(len(subpop)):
        for j in range(len(restpop)):
          if compute_dist(subpop[i],restpop[j]) < min_dist:
            min_dist = compute_dist(subpop[i],restpop[j])
      dist.append(min_dist)
    elif metric == 'max':# max distance of two points, each in the cluster, this is the equivalent to cohension metric
      max_dist = 0
      for i in range(len(subpop)):
        for j in range(len(restpop)):
          if compute_dist(subpop[i],restpop[j]) > max_dist:
            max_dist = compute_dist(subpop[i],restpop[j])
      dist.append(max_dist)

    elif metric == 'avg': # average distance of points in two clusters, equivalent to avg cohension metric 
      avg_dist = 0
      for i in range(len(subpop)):
        for j in range(len(restpop)):
          avg_dist += compute_dist(subpop[i],restpop[j])
      avg_dist = avg_dist/(len(subpop)*len(restpop))
      dist.append(avg_dist)
  return dist

# count frequency of each item in a list
def CountFrequency(my_list): 
  freq = {} 
  for item in my_list: 
    if (item in freq): 
      freq[item] += 1
    else: 
      freq[item] = 1
  return freq