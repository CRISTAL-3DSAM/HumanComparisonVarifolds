from pykeops.torch import LazyTensor
import h5py
import os
import numpy as np
from tqdm import tqdm
import torch
from surfaces import Surface
import argparse
from data_parsed import *
from pathlib import Path
import sys
from collections.abc import Iterable

def list_mouv_files(mouv_path):
    """
    Helper to preload files of a CVSSP3D motion
    :str mouv_path: folder of a CVSS3D motion
    :return: list of files
    """
    files = [f for f in os.listdir(mouv_path) if ".tri" in f or ".ntri" in f]
    return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x))))


def computeNormalsCenters(vertices, faces):
    """
    Function that computes centers and surface elements (area * normal) of triangles of a mesh
    :np.array, Nvx3 vertices: vertices of the mesh
    :np.array, NFx3 faces: faces of the mesh
    :return np.array, NFx3 centers np.array , NFx3 surfel: centers and surface elements of the mesh
    """
    xDef1 = vertices[faces[:, 0], :]
    xDef2 = vertices[faces[:, 1], :]
    xDef3 = vertices[faces[:, 2], :]
    centers = (xDef1 + xDef2 + xDef3) / 3
    surfel = np.cross(xDef2 - xDef1, xDef3 - xDef1)
    areas = np.linalg.norm(surfel, axis=-1, keepdims=True)
    normals = surfel/(areas + 1e-21)
    return centers, normals, np.squeeze(areas)

device = 'cuda:0'


def lazy_current_dist_batch(areas1, normals1, centers1, areas2, normals2, centers2, sigma):
    """
    Pykeops function to compute current product over 2 batches of meshes (center + surface elements)
    :np.array, N_1x3 surfels1: surfels of batch number 1
    :np.array, N_1x3 centers1: centers of batch number 1
    :np.array, N_2x3 surfels2: surfels of batch number 2
    :np.array, N_2x3 centers2: centers of batch number 2
    :float sigma: sigma parameter of gaussian kernel
    :return np.array N_1xN_2 batch of current products:
    """
    # Current, and absolute varifolds can be defined directly from surface elements (normals * areas)
    surfels1 = areas1[:, :, None]*normals1
    surfels2 = areas2[:, :, None]*normals2
    # Transforming into Pykeops lazytensors (no direct evaluation, compute during compilation)
    surfe_laz_1 = LazyTensor(surfels1[:, None, :, None, :].contiguous().to(device))
    center_laz_1 = LazyTensor(centers1[:, None, :, None, :].contiguous().to(device))
    surfe_laz_2 = LazyTensor(surfels2[None, :, None, :, :].contiguous().to(device))
    center_laz_2 = LazyTensor(centers2[None, :, None, :, :].contiguous().to(device))
    # Computing current product
    gamma = 1 / (sigma * sigma)
    D2 = ((center_laz_1 - center_laz_2) ** 2).sum()
    K = (-D2 * gamma).exp() * (surfe_laz_1 * surfe_laz_2).sum()
    final_K = K.sum_reduction(dim=3)
    return final_K.sum(dim=[2,3])#/(surfel1.shape[0]*surfel2.shape[0])


def lazy_varifold_dist_batch(areas1, normals1, centers1, areas2, normals2, centers2, sigma):
    """
    Pykeops function to compute absolute varifolds product over 2 batches of meshes (center + surface elements)
    :np.array, N_1x3 surfels1: surfels of batch number 1
    :np.array, N_1x3 centers1: centers of batch number 1
    :np.array, N_2x3 surfels2: surfels of batch number 2
    :np.array, N_2x3 centers2: centers of batch number 2
    :float sigma: sigma parameter of gaussian kernel
    :return np.array N_1xN_2 batch of absolute products:
    """
    # Current, and absolute varifolds can be defined directly from surface elements (normals * areas)
    surfels1 = areas1[:, :, None]*normals1
    surfels2 = areas2[:, :, None]*normals2
    # Transforming into Pykeops lazytensors (no direct evaluation, compute during compilation)
    surfe_laz_1 = LazyTensor(surfels1[:, None, :, None, :].contiguous().to(device))
    center_laz_1 = LazyTensor(centers1[:, None, :, None, :].contiguous().to(device))
    surfe_laz_2 = LazyTensor(surfels2[None, :, None, :, :].contiguous().to(device))
    center_laz_2 = LazyTensor(centers2[None, :, None, :, :].contiguous().to(device))
    # Computing absolute varifolds product
    gamma = 1 / (sigma * sigma)
    D2 = ((center_laz_1 - center_laz_2) ** 2).sum()
    K = (-D2 * gamma).exp() * (surfe_laz_1 * surfe_laz_2).sum().abs()
    final_K = K.sum_reduction(dim=3)
    return final_K.sum(dim=[2,3])


def lazy_varifold_or_dist_batch(areas1, normals1, centers1, areas2, normals2, centers2, sigma):
    """
    Pykeops function to compute oriented varifolds product over 2 batches of meshes (center + surface elements)
    :np.array, N_1x3 surfels1: surfels of batch number 1
    :np.array, N_1x3 centers1: centers of batch number 1
    :np.array, N_2x3 surfels2: surfels of batch number 2
    :np.array, N_2x3 centers2: centers of batch number 2
    :float sigma: sigma parameter of gaussian kernel
    :return np.array N_1xN_2 batch of oriented varifolds products:
    """
    # Transforming into Pykeops lazytensors (no direct evaluation, compute during compilation)
    area_laz_1 = LazyTensor(areas1[:, None, :, None, None].contiguous().to(device))
    norm_laz_1 = LazyTensor(normals1[:, None, :, None, :].contiguous().to(device))
    center_laz_1 = LazyTensor(centers1[:, None, :, None, :].contiguous().to(device))
    area_laz_2 = LazyTensor(areas2[None, :, None, :, None].contiguous().to(device))
    norm_laz_2 = LazyTensor(normals2[None, :, None, :, :].contiguous().to(device))
    center_laz_2 = LazyTensor(centers2[None, :, None, :, :].contiguous().to(device))
    # Computing oriented varifolds product
    gamma = 1 / (sigma * sigma)
    D2 = ((center_laz_1 - center_laz_2) ** 2).sum()
    K = (-D2 * gamma).exp() * ((norm_laz_1 * norm_laz_2).sum()/0.5).exp() * (area_laz_1* area_laz_2).sum()
    final_K = K.sum_reduction(dim=3)
    return final_K.sum(dim=[2,3])


def compute_feats(vertices, faces, do_cm=False):
    """
    Computing centers and surface element of a mesh, converting them to Pytorch tensors. Centroid normalization if needed
    :np.array Nvx3 vertices: vertices of the mesh
    :np.array Nfx3 faces: faces of the mesh
    :bool do_cm: centroid normalization or not
    :return (torch.Tensor, Nfx3), (torch.Tensor, Nfx3), (np.array, 3): centers tensor, surfel tensor, centroid (None if not needed)
    """
    centers, normals, areas = computeNormalsCenters(vertices, faces)
    cm = None
    if do_cm:
        cm = np.average(centers, weights=areas, axis=0)
        centers = centers-cm
    #surfel = surfel/np.linalg.norm(surfel, axis=1, keepdims=True)
    return torch.from_numpy(centers).float(), torch.from_numpy(normals).float(), torch.from_numpy(areas).float(), cm



def prepare_batch(vertices_list, faces_list, centroid=False):
    """
    Computing batched centers and surface element of a list of meshes, converting them to Pytorch tensors. Centroid normalization if needed
    :np.array NbxNvx3 vertices: Batched vertices of the meshes
    :np.array NbxNfx3 faces: Batched faces of the meshes
    :bool do_cm: centroid normalization or not
    :return (torch.Tensor, NbxNfx3), (torch.Tensor, NbxNfx3), (np.array, Nbx3): batched centers tensor, surfel tensor, centroid (None if not needed)
    """
    Ns = [f.shape[0] for f in faces_list]
    N = max(Ns)
    batch_areas = torch.zeros((len(faces_list), N))
    batch_normals = torch.zeros((len(faces_list), N, 3))
    batch_centers = torch.zeros((len(faces_list), N, 3))
    centroids = []
    for i in range(len(faces_list)):
        N_i = faces_list[i].shape[0]
        batch_centers[i, :N_i, :], batch_normals[i, :N_i, :], batch_areas[i, :N_i], cm = compute_feats(vertices_list[i], faces_list[i], centroid)
        centroids.append(cm)
    if not centroid:
        centroids = None
    return batch_areas, batch_normals, batch_centers, centroids


def compute_prods_batch(mouv, sigma, centroid=False, lazy_dist_batch=lazy_current_dist_batch):
    """
    Computing inner products over a motion, using the batched product defined above. Use it when the files are stored
    in a suitable format (dyna for example).
    :list, Nm mouv: meshes (vertices, faces) of the motion
    :float sigma: sigma of the gaussian kernel
    :bool centroid: centroid normalization or not
    :function lazy_dist_batch: which kernel product, possible values:
    lazy_current_dist_batch lazy_varifold_dist_batch lazy_varifold_or_dist_batch
    :return (torch.Tensor, NmxNm), (torch.Tensor, Nmx3), : products tensor, centroids tensor (None if not needed)
    """
    len_mouv = len(mouv["vertices"])
    prods = np.zeros((len_mouv, len_mouv))
    b_size = 100
    cents = None
    if centroid:
        cents = []
    for i in tqdm(range(0, len_mouv, b_size)):
        areas_batch_i, normals_batch_i, centers_batch_i, centroids_batch = prepare_batch(mouv["vertices"][i:b_size + i], mouv["faces"][i:b_size + i], centroid)
        if centroid:
            cents += centroids_batch
        for j in range(i, len_mouv, b_size):
            areas_batch_j, normals_batch_j, centers_batch_j, _ = prepare_batch(mouv["vertices"][j:b_size+j], mouv["faces"][j:b_size+j], centroid)
            res = lazy_dist_batch(areas_batch_i, normals_batch_i, centers_batch_i,
                                  areas_batch_j, normals_batch_j, centers_batch_j, sigma).detach().cpu().numpy()
            prods[i:b_size+i, j:b_size+j] = res
            prods[j:b_size+j, i:b_size+i] = res.T
    return prods, cents


def prepare_batch_file(surface_list):
    """
    Computing batched centers and surface element of a list of meshes, converting them to Pytorch tensors. Centroid normalization if needed
    :list(Surface), Nb surface_list: List of the batched meshes in the Surface object format (see surfaces.py)
    :bool do_cm: centroid normalization or not
    :return (torch.Tensor, NbxNfx3), (torch.Tensor, NbxNfx3), (np.array, Nbx3): batched centers tensor, surfel tensor, centroid (None if not needed)
    """
    Ns = [surface.faces.shape[0] for surface in surface_list]
    N = max(Ns)
    B = len(surface_list)
    batch_areas = torch.zeros((B, N))
    batch_normals = torch.zeros((B, N, 3))
    batch_centers = torch.zeros((B, N, 3))
    for i in range(B):
        N_i = surface_list[i].faces.shape[0]
        batch_normals[i, :N_i, :] = torch.from_numpy(surface_list[i].surfel/areas)
        batch_centers[i, :N_i, :] = torch.from_numpy(surface_list[i].centers)
        batch_areas[i, :N_i] = torch.from_numpy(areas).squeeze()
    return batch_surfel, batch_centers


def compute_prods_batch_files(mouv_files, sigma, centroid=False, lazy_dist_batch=lazy_current_dist_batch):
    """
    Computing inner products over a motion, using the batched product defined above. Use it when the files are stored
    in files instead of better format.
    :list, Nm mouv: meshes (vertices, faces) of the motion
    :float sigma: sigma of the gaussian kernel
    :bool centroid: centroid normalization or not
    :function lazy_dist_batch: which kernel product, possible values:
    lazy_current_dist_batch lazy_varifold_dist_batch lazy_varifold_or_dist_batch
    :return (torch.Tensor, NmxNm), (torch.Tensor, Nmx3), : products tensor, centroids tensor (None if not needed)
    """
    # Loading files and handling possible misloads (corrupted files)
    surfaces = []
    centroids = None
    for file in mouv_files:
        try:
            s = Surface(filename=file)
            if centroid:
                if centroids is None:
                    centroids = []
                areas = 0.5 * np.linalg.norm(s.surfel, axis=1)
                cm = np.average(s.centers, weights=areas, axis=0)
                centroids.append(cm)
                s.updateVertices(s.vertices-cm)
            surfaces.append(s)

        except:
            print("Failed file: {0}".format(file))
    # Loop of computation
    len_mouv = len(surfaces)
    prods = np.zeros((len_mouv , len_mouv))
    b_size = 100
    for i in tqdm(range(0, len_mouv, b_size)):
        areas_batch_i, normals_batch_i, centers_batch_i = prepare_batch_file(surfaces[i:b_size + i])
        for j in range(i, len_mouv, b_size):
            areas_batch_j, normals_batch_j, centers_batch_j = prepare_batch_file(surfaces[j:b_size + j])
            res = lazy_dist_batch(areas_batch_i, normals_batch_i, centers_batch_i,
                                  areas_batch_j, normals_batch_j, centers_batch_j, sigma).detach().cpu().numpy()
            prods[i:b_size+i, j:b_size+j] = res
            prods[j:b_size+j, i:b_size+i] = res.T
    return prods, centroids


def compute_database_file(data_path, res_file, sigma, centroid=False, lazy_dist_batch=lazy_current_dist_batch):
    """
    Compute the products over CVSSP3D real and synth datasets saves them to a h5 file.
    Can be (theoretically) adapted to any dataset with the same format (see readme)
    :str data_path: path of motions
    :str res_file: h5py destination file of products
    :float sigma: sigma parameter of gaussian kernel
    :bool centroid: centroid normalization or not
    :function lazy_dist_batch: which kernel product, possible values:
    lazy_current_dist_batch lazy_varifold_dist_batch lazy_varifold_or_dist_batch
    :return None:
    """
    global path_mouv
    persons = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    if os.path.exists(res_file):
        with h5py.File(res_file, 'a') as f:
            for person in persons:
                print(person)
                person_path = os.path.join(data_path, person)
                mouvs = [f for f in os.listdir(person_path) if os.path.isdir(os.path.join(data_path, person, f))]
                for mouv in mouvs:
                    print("Person: {0}, Mouvement {1}".format(person, mouv))
                    path_mouv = os.path.join(person_path, mouv)
                    key = person + "_" + mouv
                    if not (key in f):
                        mouv_files = list_mouv_files(path_mouv)
                        P, cents = compute_prods_batch_files([os.path.join(path_mouv, fi) for fi in mouv_files], sigma,
                                                             centroid, lazy_dist_batch=lazy_dist_batch)
                        f.create_dataset(key, data=P)
                        if centroid:
                            f.create_dataset(key + "_centroids", data=cents)
    else:
        with h5py.File(res_file, 'w') as f:
            for person in persons:
                print(person)
                person_path = os.path.join(data_path, person)
                mouvs = [f for f in os.listdir(person_path) if os.path.isdir(os.path.join(data_path, person, f))]
                for mouv in mouvs:
                    print("Person: {0}, Mouvement {1}".format(person, mouv))
                    path_mouv = os.path.join(person_path, mouv)
                    mouv_files = list_mouv_files(path_mouv)
                    P, cents = compute_prods_batch_files([os.path.join(path_mouv, f) for f in mouv_files], sigma,
                                                         centroid, lazy_dist_batch=lazy_dist_batch)
                    key = person + "_" + mouv
                    f.create_dataset(person + "_" + mouv, data=P)
                    if centroid:
                        f.create_dataset(key + "_centroids", data=cents)




def open_sequence(sid, seq, file):
    sidseq = sid + "_" + seq
    if sidseq not in file:
        print('Sequence %s from subject %s not in file' % (seq, sid))
        return None

    verts = file[sidseq][()].transpose([2, 0, 1])
    faces = file['faces'][()]

    return verts, faces


def compute_dyna(data_path, res_file, sigma, centroid=False, lazy_dist_batch=lazy_current_dist_batch):
    """
    Compute the products over Dyna dataset saves them to a h5 file.
    Can be (theoretically) adapted to any dataset with the same format (see readme)
    :str data_path: path of Dyna dataset
    :str res_file: h5py destination file of products
    :float sigma: sigma parameter of gaussian kernel
    :bool centroid: centroid normalization or not
    :function lazy_dist_batch: which kernel product, possible values:
    lazy_current_dist_batch lazy_varifold_dist_batch lazy_varifold_or_dist_batch
    :return None:
    """
    if os.path.exists(res_file):
        with h5py.File(res_file, "w") as f_res:
            for gender in ["dyna_dataset_m.h5", "dyna_dataset_f.h5"]:
                with h5py.File(os.path.join(data_path, gender), 'r') as f:
                    for sid in sids:
                        for seq in pids:
                            if sid + "_" + seq not in f_res:
                                op = open_sequence(sid, seq, f)
                                if op is not None:
                                    verts, faces = op
                                    time_length = verts.shape[0]
                                    mouv = {"vertices": verts, "faces": np.repeat(faces[np.newaxis, :], repeats=time_length, axis=0)}
                                    P, cents = compute_prods_batch(mouv, sigma, centroid, lazy_dist_batch=lazy_dist_batch)
                                    f_res.create_dataset(sid + "_" + seq, data=P)
                                    if centroid:
                                        f.create_dataset(sid + "_" + seq, data=cents)
    else:
        with h5py.File(res_file, "w") as f_res:
            for gender in ["dyna_dataset_m.h5", "dyna_dataset_f.h5"]:
                with h5py.File(os.path.join(data_path, gender), 'r') as f:
                    for sid in sids:
                        for seq in pids:
                            op = open_sequence(sid, seq, f)
                            if op is not None:
                                verts, faces = op
                                time_length = verts.shape[0]
                                mouv = {"vertices": verts, "faces": np.repeat(faces[np.newaxis, :], repeats=time_length, axis=0)}
                                P, cents = compute_prods_batch(mouv, sigma, centroid, lazy_dist_batch=lazy_dist_batch)
                                f_res.create_dataset(sid + "_" + seq, data=P)
                                if centroid:
                                    f.create_dataset(sid + "_" + seq, data=cents)


funcs_dict = {"current": lazy_current_dist_batch,
              "absolute": lazy_varifold_dist_batch,
              "oriented": lazy_varifold_or_dist_batch}

bdd_dict = {"real": compute_database_file,
            "synt": compute_database_file,
            "dyna": compute_dyna}

def launch_experiment(dest_path, bdd_paths, sigmas, varifold=None, cent_spec=None):
    no_cent = False
    cent = False
    if cent_spec is None:
        no_cent, cent = True, True
    elif cent_spec:
        cent = True
    else:
        no_cent = True
    if varifold is None:
        varifold = ["current", "absolute", "oriented"]
    else:
        varifold = [varifold]
    if not isinstance(sigmas, Iterable):
        sigmas = [sigmas]
    for bdd, data_path in bdd_paths.items():
        compute_func = bdd_dict[bdd]
        dest_path_bdd = os.path.join(dest_path, bdd)
        os.makedirs(dest_path_bdd, exist_ok=True)
        for name in varifold:
            for sig in sigmas:
                print("Computation on {0}, sigma = {1}".format(bdd, sig))
                ### Currents computation
                if no_cent:
                    res_file = os.path.join(dest_path_bdd, "prods_{0}_{1}.h5py".format(name, sig))
                    if not os.path.exists(res_file):
                        compute_func(data_path, res_file, float(sig), lazy_dist_batch=funcs_dict[name])
                if cent:
                    res_file = os.path.join(dest_path_bdd, "prods_{0}_centroid_{1}.h5py".format(name, sig))
                    if not os.path.exists(res_file):
                        compute_func(data_path, res_file, float(sig), centroid=True, lazy_dist_batch=funcs_dict[name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_path', dest='real_path', type=Path, required=False,
                        help="CVSSP3D Real dataset path")
    parser.add_argument('--synt_path', dest='synt_path', type=Path, required=False,
                        help="CVSSP3D artificial dataset path")
    parser.add_argument('--dyna_path', dest='dyna_path', type=Path, required=False,
                        help="Dyna dataset path")
    parser.add_argument('--dest_path', dest='dest_path', type=Path, required=False, default="",
                        help="Inner product gram martrices save folder")
    parser.add_argument('--sigma', dest='sigmas', type=float, required=False,
                        help="Sigma value. All sigmas of the paper are run if None")
    parser.add_argument('--varifold', dest='varifold', type=str, required=False,
                        help="Varifold type. current, absolute or oriented. Default all varifolds in one experience")
    parser.add_argument('--centroid', dest='centroid', type=bool, required=False,
                        help="Centroid normalization or not. Default both experience are runs")

    args = parser.parse_args(sys.argv[1:])
    bdd_paths = {}
    if args.real_path is not None:
        bdd_paths["real"] = args.real_path
    if args.synt_path is not None:
        bdd_paths["artif"] = args.synt_path
    if args.dyna_path is not None:
        bdd_paths["dyna"] = args.dyna_path
    data_dest = args.dest_path
    varifold = args.varifold
    print(varifold)
    if args.sigmas is None:
        sigmas = np.logspace(-3, 1, 10)
    else:
        sigmas = args.sigmas
    centroid = args.centroid
    launch_experiment(data_dest, bdd_paths, sigmas, varifold, centroid)

