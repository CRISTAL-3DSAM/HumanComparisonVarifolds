from model.model import SpiralEncoder, SpiralDecoder
from spiral_utils import generate_spirals, get_adj_trigs
from data.mesh_sampling import generate_transform_matrices
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import os
import torch
import json
from psbody.mesh import Mesh
from classification import print_measures, load_data_gt
from process_databases import name_to_id
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib.pyplot as plt
config_path = '/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

if config['train']['deterministic']:
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# path to downsampling/upsampling matrices
sampling_path = config['path']['sampling_matrices']
# sampling factors between two consecutive mesh resolutions
# reference SMPL template
body_model_path = ""
ref_bm_path = os.path.join(body_model_path, 'neutral/model.npz')
ref_bm = np.load(ref_bm_path)
faces = ref_bm["f"]
ref_mesh = Mesh(v=ref_bm['v_template'], f=ref_bm['f'])
scale_factors = config['model']['scale_factors']
os.makedirs(sampling_path, exist_ok=True)
if not os.path.exists(os.path.join(sampling_path, 'matrices.pkl')):
    print('Generating mesh sampling matrices')
    M, A, D, U, F = generate_transform_matrices(ref_mesh, scale_factors)

    with open(os.path.join(sampling_path, 'matrices.pkl'), 'wb') as f:
        M_vf = [(M[i].v, M[i].f) for i in range(len(M))]
        pickle.dump({'M':M_vf,'A':A,'D':D,'U':U,'F':F}, f)
else:
    print('Loading mesh sampling matrices')
    with open(os.path.join(sampling_path, 'matrices.pkl'), 'rb') as f:
        matrices = pickle.load(f)
        M = [Mesh(v=v, f=f) for v,f in matrices['M']][:len(scale_factors)+1]
        A = matrices['A'][:len(scale_factors)+1]
        D = matrices['D'][:len(scale_factors)]
        U = matrices['U'][:len(scale_factors)]
        F = matrices['F'][:len(scale_factors)]

for i in range(len(D)):
    D[i] = D[i].todense()
    U[i] = U[i].todense()

# reference vertex id when calculating spirals, check Neural3DMM for details
reference_points = [[0]]
for i in range(len(config['model']['scale_factors'])):
    dist = euclidean_distances(M[i+1].v, M[0].v[reference_points[0]])
    reference_points.append(np.argmin(dist,axis=0).tolist())
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')

adj, trigs = get_adj_trigs(A, F, ref_mesh)
spirals, spiral_sizes, _ = generate_spirals(config['model']['conv_hops'],
                                            M[:-1], adj[:-1], trigs[:-1], reference_points[:-1],
                                            dilation=config['model']['dilation'])

spirals = [torch.from_numpy(s).long().to(device) for s in spirals]
D = [torch.from_numpy(s).float().to(device) for s in D]
U = [torch.from_numpy(s).float().to(device) for s in U]

# number of feature channels for each mesh resolution
shape_enc_filters = config['model']['shape_enc_filters']
pose_enc_filters = config['model']['pose_enc_filters']
dec_filters = config['model']['dec_filters']
# dimensions for latent components
shape_dim = config['model']['shape_dim']
pose_dim = config['model']['pose_dim']
# activation function
activation = config['model']['activation']


#device = "cpu"
shape_enc = SpiralEncoder(shape_enc_filters, spirals, shape_dim, D, act=activation).to(device)#, bn=False)
pose_enc = SpiralEncoder(pose_enc_filters, spirals, pose_dim, D, act=activation).to(device)#, bn=False)
dec = SpiralDecoder(dec_filters, spirals, shape_dim+pose_dim, U, act=activation).to(device)

model_dict = {
    'shape_enc': shape_enc,
    'pose_enc': pose_enc,
    'dec': dec,
}
check_byte = torch.load("model.pth", map_location="cpu", encoding="latin1")
def decode(k):
    if isinstance(k, bytes):
        return k.decode("utf-8")
    else:
        return k
def convert_keys_to_string(dictionary):
    """Recursively converts dictionary keys to strings."""
    if not isinstance(dictionary, dict):
        return dictionary
    return dict((decode(k), convert_keys_to_string(v))
        for k, v in dictionary.items())
checkpoint = {key: convert_keys_to_string(check_byte[key]) for key in check_byte}
torch.set_rng_state(checkpoint['torch_rnd'].cpu())

checkpoint = {key: {k.replace("module.", ""): checkpoint[key][k] for k in checkpoint[key]} for key in checkpoint if "m_" in key}
for k in model_dict:
    model_dict[k].load_state_dict(checkpoint['m_' + k])#, prefix="bytes")
    model_dict[k] = model_dict[k].to(device)

for k in model_dict:
    model_dict[k].load_state_dict(checkpoint['m_' + k])#, prefix="bytes")


from shape_pose_disent.utils import SMPL2Mesh
smpl2mesh = SMPL2Mesh(bm_path=body_model_path)
gdr2num = {'male':-1, 'neutral':0, 'female':1}

from surfaces import Surface

from dyna_eval import male, female, sids, pids, open_sequence, classif_dyna
import h5py

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
                                surf = Surface(FV=[faces, verts[i]])
                                work = torch.from_numpy(surf.vertices).unsqueeze(0)
                                with torch.no_grad():
                                    pose_tr = model_dict["pose_enc"](work.float().to(device))
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
    file_res = "dyna_res/datas_dyna_uns.h5py" #No need to change it
