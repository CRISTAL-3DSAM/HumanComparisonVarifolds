import os
from collections import defaultdict
from retrieval_scores import *
from data_parsed import *
from tqdm import tqdm
from numba import jit
import h5py
import argparse
from pathlib import Path
import sys
import itertools

@jit
def sum_ij(prods, i, j, s):
    # Bin i,j of the Gram Hankel matrix
    res = 0
    for k in range(s):
        res += prods[i+k, j+k]
    return res

@jit
def compute_gram(prods, r):
    # GramHankel matrix of size r
    G = np.zeros((r, r))
    n = prods.shape[0]
    s = n-r
    for i in range(r):
        for j in range(r):
            G[i, j] = sum_ij(prods, i, j, s)
    return G/np.linalg.norm(G)


def compute_score(path, r, normalize=False, bdd="real"):
    # Compute scores of Gram Hankel matrices of size r, for a given dataset, given stored .h5py products.
    Gs = []
    mouvs_G = []
    indices = []
    with h5py.File(path, 'r') as f:
        # Quite repetitive loops, but there is some slight differences between datasets.
        if bdd == "real":
            for ind, mouv in enumerate(the_dict):
                all_occ = the_dict[mouv]
                for where in all_occ:
                    for j, index in enumerate(all_occ[where]):
                        key = pou[where][j] + "_{:03d}".format(index)
                        prods = f[key][()]
                        if normalize:
                            norms = np.sqrt(np.diag(prods))
                            prods = prods/(norms[:, None]*norms[None, :]+1e-15)
                        Gs.append(compute_gram(prods, r))
                        mouvs_G.append(classes[mouv])
                        indices.append((mouv, where, j))
        elif bdd == "artif":
            for person in persons:
                for index, mouv in enumerate(ext_mouvs):
                    key = person + "_{0}".format(mouv)
                    prods = f[key][()]
                    if normalize:
                        norms = np.sqrt(np.diag(prods))
                        prods = prods / (norms[:, None] * norms[None, :] + 1e-15)
                    Gs.append(compute_gram(prods, r))
                    mouvs_G.append(index)
                    #indices.append((mouv, where, j))
        elif bdd == "dyna":
            for sid in sids:
                for index, seq in enumerate(pids):
                    key = sid + "_" + seq
                    if key in f:
                        prods = f[key][()]
                        if normalize:
                            norms = np.sqrt(np.diag(prods))
                            prods = prods / (norms[:, None] * norms[None, :] + 1e-15)
                        Gs.append(compute_gram(prods, r))
                        mouvs_G.append(index)
    N_mous = len(Gs)
    dist_mat = np.zeros(((N_mous, N_mous)))
    for i in range(N_mous): #, "Distance matrix"):
        for j in range(i):
            dist_mat[i, j] = np.linalg.norm(Gs[i]-Gs[j])
            dist_mat[j, i] = dist_mat[i, j]
    mouv_array = np.array(mouvs_G)
    scores, _, _, _ = print_measures(mouv_array, dist_mat)
    return scores


def print_all_scores(path, normalize=False, centroid=False, varifolds="current", bdd="real", rmax=50):
    # Compute scores of Gram Hankel matrices overall possible sizes r, for a given dataset, given stored .h5py products.
    # Better use the print save score than this function!
    R = np.arange(rmax) +1
    files = [f for f in os.listdir(path) if ".h5py" in f]
    work_files = None
    if varifolds is not None:
        work_files = [f for f in files if varifolds in f]
    else:
        work_files = [f for f in files if (len(f.split("_")) ==2)]
        work_files += [f for f in files if (len(f.split("_")) ==3) and "centroid" in f]
    if centroid:
        work_files = [f for f in work_files if ("centroid" in f)]
    else:
        work_files = [f for f in work_files if not ("centroid" in f)]
    print(work_files)
    sigmas = [float(".".join(file.split(".")[:-1]).split("_")[-1]) for file in work_files]
    sig_file = zip(sigmas, work_files)
    sig_file = sorted(sig_file, key=lambda x: x[0])
    result = defaultdict(dict)
    best_params = None
    best_nn, best_ft, best_st = 0, 0, 0
    best_s = None
    for r in tqdm(R):
        for sigma, file in sig_file:
            full = os.path.join(path, file)
            s = compute_score(full, r, normalize, bdd=bdd)
            if s[0]> best_nn:
                best_nn, best_ft, best_st = s
                best_s = s
                best_params = (sigma, r)
            elif s[1] > best_ft and s[0] == best_nn:
                best_nn, best_ft, best_st = s
                best_s = s
                best_params = (sigma, r)
            elif s[2] > best_st and s[0] == best_nn and s[1] == best_ft:
                best_nn, best_ft, best_st = s
                best_s = s
                best_params = (sigma, r)
            result[sigma][r] = s
    print("Best score: {0}, params: sigma={1}, r={2}".format(best_s, best_params[0], best_params[1]))
    return best_nn, best_params, result


def print_save_scores(path_scores, path, norm=False, centroid=False, varifolds=None, bdd="real", rmax=50):
    '''
    Function that compute scores over all sigma values of a given configuration
    :str path_scores: The path where to store the computed scores in npy format
    :str path: The path where the .h5py gram matrix of each motion are stored
    :bool norm: Whether to use product normalization
    :bool centroid: Whether to use centroid normalization (computed before)
    :str varifolds: set to None if current, or for oriented varifolds, "abs" for absolute varifolds
    :str bdd: "real" for CVSSP3D real dataset, "artif" for artificial, "dyna" for Dyna dataset
    :int rmax: Max R value for Gram-Hankel matrix. R=50 for real, 100 for artificial, and 140 for Dyna datasets
    :return: None (prints the best score)
    '''
    centr_str = "_centroid" if centroid else ""
    norm_str = "_norm" if norm else ""
    print(varifolds)
    vari_str = "" if varifolds is None else "_" + varifolds
    full_path = os.path.join(path_scores, "score{0}{1}{2}.npy".format(norm_str, vari_str, centr_str))
    print(full_path)
    if not os.path.exists(full_path):
            s1 = print_all_scores(path, norm, centroid=centroid, varifolds=varifolds, bdd=bdd, rmax=rmax)
            np.save(full_path, s1)
    else:
        _, params, scores = np.load(full_path, allow_pickle=True)
        print("Best score: {0}, params: sigma={1}, r={2}".format(scores[params[0]][params[1]], params[0], params[1]))


rmax_dicts = {"real": 50, "artif": 100, "dyna": 140}  # Maximum r value for Gram-Hankel matrices by default.


def init_cent_norm(norm_spec, cent_spec):
    """
    Helper function to choose all centroid options and inner product options
    :param norm_spec: normalization spec of the inner product
    :param cent_spec: normalization spec of the centroid
    :return: loop of all possible options
    """
    loop_cent = []
    if cent_spec is None:
        loop_cent.append(True)
        loop_cent.append(False)
    elif cent_spec:
        loop_cent.append(True)
    else:
        loop_cent.append(False)
    loop_norm = []
    if norm_spec is None:
        loop_norm.append(True)
        loop_norm.append(False)
    elif norm_spec:
        loop_norm.append(True)
    else:
        loop_norm.append(False)
    loop = list(itertools.product(*[loop_cent, loop_norm]))  # Cartesian product of options
    return loop


def full_print(path_scores, bdd_paths, norm_spec=None, cent_spec=None, varifolds=None, rmax=None):
    """
    Function that computes cross validated scores
    :str path_scores: where to save scores
    :dict(str) bdd_paths: where are the h5pys
    :bool norm_spec: which inner product normalization (default to None, means both)
    :bool cent_spec: which centroid normalization (default to None, means both)
    :list[str] varifolds: which varifolds to compare (default to None, means all -current, oriented, absolute)
    :int rmax: Maximum r value of Gram Hankel matrix
    :return None:
    """
    option_loop = init_cent_norm(norm_spec, cent_spec)
    for bdd, path in bdd_paths.items():
        path_scores_bdd = os.path.join(path_scores, bdd)
        os.makedirs(path_scores_bdd, exist_ok=True)
        for varifold_type in varifolds:
            if rmax is None:
                R = rmax_dicts[bdd]
            else:
                R = rmax
            for cent_opt, norm_opt in option_loop:
                print_save_scores(path_scores_bdd, path, norm_opt, cent_opt, varifold_type, bdd, rmax=R)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_path', dest='real_path', type=Path, required=False,
                        help="CVSSP3D Real dataset inner product gram matrices path")
    parser.add_argument('--synt_path', dest='synt_path', type=Path, required=False,
                        help="CVSSP3D artificial dataset inner product gram matrices path")
    parser.add_argument('--dyna_path', dest='dyna_path', type=Path, required=False,
                        help="Dyna dataset inner product path")
    parser.add_argument('--dest_path', dest='dest_path', type=Path, required=False, default="",
                        help="Score npys save folder")
    parser.add_argument('--varifold', dest='varifold', type=str, required=False,
                        help="Varifold type. current, absolute or oriented. Default all varifolds in one experience")
    parser.add_argument('--inner_norm', dest='norm', type=bool, required=False,
                        help="Inner product normalization or not. Default both experience are runs")
    parser.add_argument('--centroid', dest='centroid', type=bool, required=False,
                        help="Centroid normalization or not. Default both experience are runs")
    parser.add_argument('--rmax', dest='rmax', type=int, required=False,
                        help="Maximum R value for the experience. Default depends on dataset (see line 154)")

    args = parser.parse_args(sys.argv[1:])
    bdd_paths = {}
    if args.real_path is not None:
        bdd_paths["real"] = args.real_path
    if args.synt_path is not None:
        bdd_paths["artif"] = args.synt_path
    if args.dyna_path is not None:
        bdd_paths["dyna"] = args.dyna_path
    print(bdd_paths)
    path_scores = args.dest_path
    varifold = args.varifold
    if varifold is None:
        varifold = ["current", "absolute", "oriented"]
    centroid = args.centroid
    norm = args.norm
    rmax = args.rmax
    full_print(path_scores, bdd_paths, norm_spec=norm, cent_spec=centroid, varifolds=varifold, rmax=rmax)