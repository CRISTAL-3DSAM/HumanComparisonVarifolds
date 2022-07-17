import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import scipy.sparse as sp
import torch 
import dill
import pickle
import torch.nn as nn
import shutil
from tqdm import tqdm
import trimesh 
from models.model import PointNetVAE
import numpy as np
import h5py

device = 'cpu'



#model options
opt = lambda x:x
opt.NUM_POINTS = 2100
opt.BATCH_SIZE = 16
opt.DESC_SIZE = 512 #pointwise descriptro size after convlolutional layers
opt.LATENT_SPACE = 256 #dimension of the full (pose+style) latent space
opt.POSE_SIZE = 64 #number of dimension dedicated to pose encoding 

opt.LOCAL_TH = 0.1
opt.LEARNING_RATE = 0.1e-4

vae = PointNetVAE(opt).to(device)


#load pretrained model
net_type = '4'

loss_step = '' 
loss_step = '_ae_euc' 
loss_step = '_ae_euc_gd1' 
loss_step = '_ae_euc_gd2' 

vae.load_state_dict(torch.load('pretrained/FAUST_vae_euc_gd.dict'), strict=False)
vae.eval()

def gather_pose(vertices):
    lsp = vae.enc(vertices)
    pose = lsp[:,opt.POSE_SIZE:opt.LATENT_SPACE]
    return pose

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


def dyna_compute(path, output):
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
                                surf = trimesh.base.Trimesh(vertices=verts[i], faces=faces)
                                vertices = surf.vertices
                                vertices[:, 1] -=0.85+np.min(vertices[:,1])
                                work = torch.from_numpy(vertices).unsqueeze(0)
                                with torch.no_grad():
                                    pose_tr = gather_pose(work.float().to(device))
                                    pose = pose_tr.detach().cpu().squeeze().numpy()
                                descs.append(pose)
                            descs = np.array(descs)
                            f_res.create_dataset(sid+"_"+seq, data=descs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dyna_path', dest='dyna_path', type=Path, required=True,
                        help="Dyna dataset location path")
	args = parser.parse_args(sys.argv[1:])
    dyna_path = args.dyna_path
	res_file = "dyna_res/data_dyna_limp.h5py"
	dyna_compute(dyna_path, res_file)
