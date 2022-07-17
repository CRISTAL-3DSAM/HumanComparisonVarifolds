# Code for ECCV 2022 submission number 6963

## Generating inner products over a motion (gram matrices).
The code to generate gram hankel matrices can be found in gram_varifolds_motion.py. 
You will need to set properly the folder of each dataset in real_path, dyna_path and synt_path.
The dest_path will be set by default to current folder.
The full experience (all sigmas, centroid or not) will run over 2 weeks for a single GPU. 
The h5pys are available here https://drive.google.com/file/d/18qum9qabgvweYETWxPIr_52TkHEFgP5o/view?usp=sharing
```commandline
python gram_varifolds_motion.py --real_path="/path/to/cvssp3d/real" --synt_path="/path/to/cvssp3d/artif" --dyna_path="/path/to/dyna" --dest_path="/path/to/save"
```
To compute a single experience, and set manually the sigmas, run the script in the following way (dyna_path for dyna, 
real_path for CVSSP3D real, synt_path for CVSSP3D artificial).
```commandline
python gram_varifolds_motion.py --dyna_path="/path/to/dyna"  --varifold="absolute" --sigma=0.4 --centroid=true --dest_path="/path/to/save" 
```
The command below set the script to run the absolute varifolds experience (other options, "current" and "oriented"), with
gaussian kernel sigma set to 0.4, and centroid normalization (false to avoid normalization). 
## Computing Gram Hankel matrices and their associated score
The code to compute the score associated to Hankel matrices can be found in gram_hankel.py. This script will compute 
the Hankel matrices for r values between 1 and maximum r value, and will print the final score. The computed scores are 
save in npy files. You will need to set properly the folder of each  dataset experiments (stored h5pys). 
Each folder must contain the inner products of only one dataset (see google drive formats). 
The dest_path will be set by default to current folder. 
The full experience (all dataset, all normalizations, all varifolds type) will run for several hours on a modern CPU. 
This experience doesn't need Pykeops, and can be run on Windows. 
```commandline
python gram_hankel.py --real_path="/path/to/cvssp3d/real/h5pys" --synt_path="/path/to/cvssp3d/artif/h5pys" --dyna_path="/path/to/dyna/h5pys" --dest_path="/path/to/save/scores" 
```
To compute a single experience, and set manually the sigmas, run the script in the following way (dyna_path for dyna, 
real_path for CVSSP3D real, synt_path for CVSSP3D artificial).
```commandline
python gram_hankel.py --dyna_path="/path/to/dyna" --varifold="absolute" --centroid=True --inner_norm=True --rmax=42
```
The command below set the script to run the absolute varifolds experience (other options, "current" and "oriented"), with
centroid normalization and inner product normalization. The maximum r value for Gram-Hankel matrices is set to 42.
## Other files
Those files are helper files that should not be loaded, but are used to compute files.
- retrieval_scores.py : computes the score of each experience
- surfaces.py : Helper class to load CVSSP3D mesh files

## Comparisons
The scripts to launch comparisons with GDVAE, LIMP, Zhou et al, Areas and Breadths and are available in the comparisons 
folder. For the first three, a script to generate the features of the motions are provided.
The user will need to launch the scripts in the folder of the code of each method:
- https://gitlab.com/taumen/gdvae
- https://github.com/lcosmo/LIMP
- https://github.com/kzhou23/shape_pose_disent

Once the h5py are generated (available in comparisons/dyna_res anyway!), the script comparisons.py can be launched. 
Be careful to put the files in "dyna_res" folder inside comparisons folder. We provide the precomputed h5py for each method.
The script comparisons.py can be launched as follow:
```commandline
python comparisons.py --dyna_path="/path/to/dyna"
```
To add comparisons with deep learning methods (here just limp, but you can add gdvae and uns in the same manner):
```commandline
python comparisons.py --dyna_path="/path/to/dyna" --limp=true
```
The full script should run less than one hour since no cross validation is needed.

## Requirements 
The pykeops library is only usable in Linux. The scores computation can be made using the precomputed provided h5py.
We recommend to use anaconda, and to create an environment with installed packages:
```commandline
conda env create -f environment.yml
conda activate ECCV_2022_6963
```
In case you want to install yourself in a custom way, the main requirements are:
- Pytorch (run experience varifolds + deep learning experiments)
- Numpy
- Pykeops (run experience varifolds)
- Scipy
- Scikit-learn
- tslearn
- tqdm

Without Pytorch and Pykeops, you should still be able to compute scores.

## Dataset organisation

In case you would like to compute the whole h5pys inner product, you will find the datasets as follow:
- CVSSP3D artificial : you will need to contact Adrien Hilton and ask him for data : a.hilton@surrey.ac.uk. 
Please ask for "cvssp3d" dataset.
Once downloaded, simply unzip all individual folders in the same folder, organized as follow: artif/person/motion/surface.tri
- CVSSP3D real you will need to contact Adrien Hilton and ask him for data : a.hilton@surrey.ac.uk. 
Please ask for "i3d_pose_action" dataset
Once downloaded, simply unzip all motion folders in folders, organized as follow: artif/person/motion_id/surface.ntri
where motion_id is the number of the motion when downloading
- Dyna dataset: download the dataset here http://dyna.is.tue.mpg.de/. Please put the 2 h5 files 
(dyna_dataset_f.h5, dyna_dataset_m.h5) in the same folder
