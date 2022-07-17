import numpy as np

def print_measures(positions, mat):
    """
    Function that prints NN, FT and ST score, and return individuals score for each sample along with FT query
    :np.array(N) positions: class of each element
    :np.array, NxN mat: distance matrix of the dataset
    :return tuple(float, float, float) scores, np.array(N) NN scores, np.array(N) FT scores, np.array(N) ST scores:
    """
    my_tier = full_tier_mat(mat, positions)
    NN, NN_array = nearest(my_tier, positions)
    FT, FT_array = first_tier(my_tier, positions)
    ST, ST_array = second_tier(my_tier, positions)
    score = (NN, FT, ST)
    return score, NN_array, FT_array, ST_array


def full_tier_mat(dist_mat, positions):
    """
    Function that executes retrieval query
    :np.array NxN dist_mat: distance matrix of the dataset
    :np.array(N) positions: class of each element
    :return full tier: full second tier of the query
    """
    result = ["" for i in range(dist_mat.shape[0])]
    N = dist_mat.shape[0]
    for i in range(N):
        pose = positions[i]
        tier = (positions==pose).sum()-1
        bests = np.argsort(dist_mat[i, :])
        closest_2_tier = bests[1:tier*2]
        result[i] = closest_2_tier
    return result


def nearest(full_tier, positions):
    """
    Function that returns nearest neighbor score of a query
    :full_tier: full second tier of the query
    :np.array(N) positions: class of each element
    :return float NN, np.array(N): NN score, NN individual scores
    """
    N = len(full_tier)
    result = np.zeros(N, dtype=np.float32)
    for i in range(N):
        pose = positions[i]
        nn = full_tier[i][0]
        result[i] = int(positions[nn] == pose)
    return result.mean(), result


def first_tier(full_tier, positions):
    """
    Function that returns second tier score of a query
    :full_tier: full second tier of the query
    :np.array(N) positions: class of each element
    :return float FT, np.array(N): FT score, FT individual scores
    """
    N = len(full_tier)
    result = np.zeros(N)
    for i in range(N):
        pose = positions[i]
        tier = full_tier[i]
        n_class = (positions == pose).sum()-1
        tier = tier[:n_class]
        result[i] = ((positions[tier] == pose).sum()*1.0) / n_class
    return result.mean(), result


def second_tier(full_tier, positions):
    """
    Function that returns second tier score of a query
    :full_tier: full second tier of the query
    :np.array(N) positions: class of each element
    :return float ST, np.array(N): ST score, ST individual scores
    """
    N = len(full_tier)
    result = np.zeros(N)
    for i in range(N):
        pose = positions[i]
        tier = full_tier[i]
        n_class = (positions == pose).sum()-1
        result[i] = ((positions[tier] == pose).sum()*1.0) / n_class
    return result.mean(), result