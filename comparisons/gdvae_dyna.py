import numpy as np
from surfaces import Surface
from tqdm import tqdm
import os
import torch

import h5py

device = "cuda:0"



VIS_SETTINGS = {
        'smal' : {
            'R' : np.array([[1.0,           0.0,              0.0],
                            [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                            [0, np.sin(np.pi/2),  np.cos(np.pi/2)]]),
            'sampling_hpad' : 0.25
        },
        'mnist' : {
            'R' : np.array([[1.0, 0,   0],
                            [0,   1.0, 0],
                            [0,   0,   1.0]]),
            'sampling_hpad' : 0.1
        },
        'smpl' : {
            'R' : np.array([[1.0, 0,   0],
                            [0,   1.0, 0],
                            [0,   0,   1.0]]),
            'sampling_hpad' : 0.1
        }
}

from tasp.projects.IEVAE.GDVAE import GDVAE
from tasp.projects.IEVAE.gdvae_autoencoder import Autoencoder_IEVAE_fixedDec

def load_models(AE_model, VAE_model):
    print('Loading AE model', AE_model)
    AE_model = Autoencoder_IEVAE_fixedDec.load(AE_model, mode='eval')
    print('Loading VAE model', VAE_model)
    VAE_model = GDVAE.load(VAE_model, mode='eval')
    print('Converting models to cpu')
    AE_model = AE_model.to(device)
    VAE_model = VAE_model.to(device)
    return AE_model, VAE_model

AE_model_path = "models/sursmpl2-031919-AE.pt"
VAE_model_path = "models/sursmpl-GDVAE-032019-T8.pt"
models = load_models(AE_model_path, VAE_model_path)

def latent_comp(AE_model, VAE_model, point_clouds):
    use_zero_for_zr_in_interps = True
    vsettings = VIS_SETTINGS["smpl"]
    rot_dim = VAE_model.rotation_dim
    ext_dim = VAE_model.extrinsic_dim
    int_dim = VAE_model.intrinsic_dim
    RE_dim = rot_dim + ext_dim
    from torch.autograd import Variable
    with torch.no_grad():
        #print(point_clouds.shape)
        L = VAE_model.encode( AE_model.encode( Variable(
                        torch.FloatTensor(point_clouds).to(device)#.unsqueeze(0)
                    ) ), return_mu_only = True )
        if use_zero_for_zr_in_interps:
           L[:, 0 : rot_dim ] = 0.0
            # Interpolate zEs and zIs
        zE= L[:, 0:RE_dim].squeeze().detach().cpu().numpy()
        zI = L[:, RE_dim:].squeeze().detach().cpu().numpy()
    return zE, zI

def representation(vertices, faces):
    #print(res_len.shape, res.shape)
    point_cloud = torch.from_numpy(vertices).float().unsqueeze(0)
    zE, _ = latent_comp(models[0], models[1], point_cloud)
    return zE


def doing_things(dat):
    i, file = dat
    try:
        return representation(*dat)
    except:
        return None


def experience(surface):
    # print(res_len.shape, res.shape)
    point_cloud = torch.from_numpy(surface.vertices).float().unsqueeze(0)
    zE, _ = latent_comp(models[0], models[1], point_cloud)
    return zE




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

def compute_desc(surface):
    # print(res_len.shape, res.shape)
    point_cloud = torch.from_numpy(surface.vertices).float().unsqueeze(0)
    zE, _ = latent_comp(models[0], models[1], point_cloud)
    return zE

def process_dyna(path, output):
    with h5py.File(output, "w") as f_res:
        for gender in [male, female]:
            with h5py.File(os.path.join(path, gender), 'r') as f:
                for sid in sids:
                    for seq in pids:
                        op = open_sequence(sid, seq, f)
                        if op is not None:
                            verts, faces = op
                            N_verts = len(verts)
                            descs = []
                            for i in tqdm(range(N_verts), "Processing {} {}".format(sid, seq)):
                                surf = Surface(FV=[faces, verts[i]])
                                descs.append(compute_desc(surf))
                            descs = np.array(descs)
                            f_res.create_dataset(sid+"_"+seq, data=descs)


if __name__ == '__main__':
    if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dyna_path', dest='dyna_path', type=Path, required=True,
                        help="Dyna dataset location path")
	args = parser.parse_args(sys.argv[1:])
    dyna_path = args.dyna_path
    file_res = "dyna_res/datas_dyna_gdvae.h5py"
    process_dyna(dyna_path, file_res)
