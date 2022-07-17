import numpy as np

import sys
sys.path.append("../")
from surfaces import Surface
import os
import h5py
from tqdm import tqdm
from tslearn.utils import to_time_series_dataset
from tslearn.metrics import dtw
from retrieval_scores import print_measures
import argparse
from pathlib import Path
import sys
from multiprocessing import Pool


def run_imap_multiprocessing(func, argument_list, num_processes):

    pool = Pool(processes=num_processes)

    result_list_tqdm = []
    for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
        if result is not None:
            result_list_tqdm.append(result)

    return result_list_tqdm


def run_linear(func, argument_list):

    result_list_tqdm = []
    for arg in tqdm(argument_list):
        result = func(arg)
        if result is not None:
            result_list_tqdm.append(result)

    return result_list_tqdm


def compute_dtw(tss):
    i, j, ts1, ts2 = tss
    dist_temp = dtw(ts1, ts2)#, global_constraint="sakoe_chiba", sakoe_chiba_radius=radius)
    return i, j, dist_temp


def compute_fast_dtw(ts_list, name_comp=""):
    N = len(ts_list)
    mat = np.zeros((N, N))
    paths = np.empty((N, N), dtype=object)
    to_compute = []
    for i in tqdm(range(N), name_comp):
        for j in range(i+1, N):
            to_compute.append((i, j, ts_list[i], ts_list[j]))
    #result = run_imap_multiprocessing(compute_dtw, to_compute, 20)
    result = run_linear(compute_dtw, to_compute)
    for i, j, dist_temp in result:
        mat[i, j] = dist_temp
        mat[j, i] = dist_temp
        #paths[i, j] = path_temp
    return mat, paths


sids = ['50004', '50020', '50021', '50022', '50025',
            '50002', '50007', '50009', '50026', '50027']
pids = ['hips', 'knees', 'light_hopping_stiff', 'light_hopping_loose',
            'jiggle_on_toes', 'one_leg_loose', 'shake_arms', 'chicken_wings',
            'punching', 'shake_shoulders', 'shake_hips', 'jumping_jacks',
            'one_leg_jump', 'running_on_spot']

male = "dyna_dataset_f.h5"
female = "dyna_dataset_m.h5"


def open_sequence(sid, seq, file):
    sidseq = sid + "_" + seq
    if sidseq not in file:
        print('Sequence %s from subject %s not in file' % (seq, sid))
        return None

    verts = file[sidseq][()].transpose([2, 0, 1])
    faces = file['faces'][()]

    return verts, faces


def computeSurfels(vertices, faces):
    """
    Function that computes surface elements (area * normal) of triangles of a mesh
    :np.array, Nvx3 vertices: vertices of the mesh
    :np.array, NFx3 faces: faces of the mesh
    :return np.array, NFx3 surfel: surface elements of the mesh
    """
    xDef1 = vertices[faces[:, 0], :]
    xDef2 = vertices[faces[:, 1], :]
    xDef3 = vertices[faces[:, 2], :]
    surfel = np.cross(xDef2 - xDef1, xDef3 - xDef1)
    return surfel


def compute_desc(vertices, faces, name_exp="areas"):
    surfel = computeSurfels(vertices, faces)
    if name_exp == "areas":
        temp = np.abs(np.inner(surfel, vectors))
        res = temp.sum(axis=0)
    elif name_exp == "breadths":
        temp = np.inner(vertices, vectors)
        res = (np.amax(temp, axis=0) - np.amin(temp, axis=0))
    return res/np.linalg.norm(res)


def process_dyna(path, output, names_exp, N):
    if not os.path.exists(output):
        with h5py.File(output, "w") as f_res:
            for gender in [male, female]:
                with h5py.File(os.path.join(path, gender), 'r') as f:
                    for sid in sids:
                        for seq in pids:
                            op = open_sequence(sid, seq, f)
                            if op is not None:
                                verts, faces = op
                                N_verts = len(verts)
                                descs = np.zeros((N_verts, 2*N))
                                for i in tqdm(range(N_verts), "Processing {} {}".format(sid, seq)):
                                    res_list = []
                                    for name in names_exp:
                                        res_list.append(compute_desc(verts[i], faces, name))
                                    descs[i, :] = np.hstack(res_list)
                                f_res.create_dataset(sid+"_"+seq, data=descs)


def all_desc_dyna(path, output_folder, names, N):
    global vectors, sqrt
    sqrt = int(np.sqrt(N))
    teta = (np.pi / sqrt) * np.arange(0, sqrt)
    # teta = np.pi * np.linspace(0, 1, N_show)
    phi = (2 * np.pi / sqrt) * np.arange(0, sqrt)
    phi_2d, theta_2d = np.meshgrid(phi, teta)

    vectors = np.array([(np.sin(theta_2d) * np.cos(phi_2d)).ravel(),
                        (np.sin(theta_2d) * np.sin(phi_2d)).ravel(),
                        np.cos(theta_2d).ravel()]).swapaxes(1, 0)
    file_name = "datas_dyna_{0}_{1}.h5py".format(N, "_".join(names))
    process_dyna(path, os.path.join(output_folder, file_name), names, N)


def result_dyna(filename, name_exp, N):
    ts_list = []
    ids = []
    with h5py.File(filename, "r") as f_res:
        for sid in sids:
            for id, seq in enumerate(pids):
                if sid + "_" + seq in f_res:
                    ts = f_res[sid + "_" + seq][()]
                    descs_ts = None
                    if name_exp == "areas":
                        descs_ts = ts[:, :N]
                    elif name_exp == "breadths":
                        descs_ts = ts[:, N:]
                    else:
                        descs_ts = ts.reshape((ts.shape[0], -1))
                    ts_list.append(descs_ts)
                    ids.append(id)
    ts_dat = to_time_series_dataset(ts_list)
    return ts_dat, ids


def classif_dyna(filename, name_exp, N=0):
    dataset_list, mouvs = result_dyna(filename, name_exp, N)
    mouvs = np.array(mouvs)
    mat, _ = compute_fast_dtw(dataset_list, name_exp)
    scores, _, _, _ = print_measures(mouvs, mat)
    print(scores)


vectors = None
sqrt = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dyna_path', dest='dyna_path', type=Path, required=True,
                        help="Dyna dataset location path")
    parser.add_argument('--limp', dest='limp_run', type=bool, default=False,
                        help="LIMP comparison. Needs a pre-h5py file")
    parser.add_argument('--uns', dest='uns_run', type=bool, default=False,
                        help="Zhou et al comparison. Needs a pre-h5py file")
    parser.add_argument('--gdvae', dest='gdvae_run', type=bool, default=False,
                        help="GDVAE comparison. Needs a pre-h5py file")
    args = parser.parse_args(sys.argv[1:])
    dyna_path = args.dyna_path
    output_path = "dyna_res" # Folder where to save experiments
    os.makedirs(output_path, exist_ok=True)
    # Areas and Breadths experimetns
    all_desc_dyna(dyna_path, output_path, ["areas", "breadths"], 64)
    classif_dyna(os.path.join(output_path, "datas_dyna_64_areas_breadths.h5py"), "areas", 64)
    classif_dyna(os.path.join(output_path, "datas_dyna_64_areas_breadths.h5py"), "breadths", 64)
    classif_dyna(os.path.join(output_path, "datas_dyna_64_areas_breadths.h5py"), "both", 64)
    # Deep learning experiments, need to be launched before
    if args.limp_run:
        classif_dyna(os.path.join(output_path, "data_dyna_limp.h5py"), "limp")
    if args.uns_run:
        classif_dyna(os.path.join(output_path, "data_dyna_uns.h5py"), "uns")
    if args.gdvae_run:
        classif_dyna(os.path.join(output_path, "data_dyna_gdvae.h5py"), "gdvae")
    print('END')