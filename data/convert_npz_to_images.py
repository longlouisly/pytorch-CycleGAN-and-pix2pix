import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from matlab_utils import keyboard

def normalize(x,r=0.5):
    # sqrt function start at 0 and is 1 at x = A
    A = np.pi*r*r  # area of the viewing circle
    return np.sqrt(x/A)


if __name__ == '__main__':
    source = sys.argv[1]
    target = sys.argv[2] 

    files = os.listdir(source)
    files.sort()

    for f in files:
        prefix = f[:-4]
        saved = np.load(os.path.join(source,f))
        psi = saved['psi']
        phi = saved['phi']
        horizons = saved['horizons']
        E = saved['E']
        E = normalize(E)


        x0 = saved['x0'] 
        m = E.shape[0]
        x0 = (x0 * m).astype(int)
        X = np.zeros((m,m,1))
        X[x0[:,0],x0[:,1]] = 1

        vis = np.expand_dims(1.0*(psi>0), -1)
        scene = np.expand_dims(1.0*(phi>0), -1)
        horizons = np.expand_dims(horizons, -1)
        E = np.expand_dims(E, -1)
        zero = np.zeros((m,m,1)) 

        # exploration: psi, hor to g
        exp_input = np.concatenate((vis, horizons, zero),axis=-1)
        exp_output = np.concatenate((E,E,E),axis=-1)
        exp_pair = np.concatenate((exp_input,exp_output),axis=1)
        exp_pair = Image.fromarray( (255*exp_pair).astype('uint8'), mode='RGB')
        exp_pair.save(os.path.join(target, 'exploration', prefix + 'exp.png'),'png')        

        # surveillance: psi, hor, phi to g
        surv_input = np.concatenate((vis, horizons, scene),axis=-1)
        surv_output = np.concatenate((E,E,E),axis=-1)
        surv_pair = np.concatenate((surv_input,surv_output),axis=1)
        surv_pair = Image.fromarray( (255*surv_pair).astype('uint8'), mode='RGB')
        surv_pair.save(os.path.join(target, 'surveillance', prefix + 'surv.png'),'png')        

        # reconstruction: psi, hor to phi
        rec_input = np.concatenate((vis, horizons, zero),axis=-1)
        rec_output = np.concatenate((scene,scene,scene),axis=-1)
        rec_pair = np.concatenate((rec_input,rec_output),axis=1)
        rec_pair = Image.fromarray( (255*rec_pair).astype('uint8'), mode='RGB')
        rec_pair.save(os.path.join(target, 'reconstruction', prefix + 'rec.png'),'png')        

        # inversion: psi, hor, g to phi
        inv_input = np.concatenate((vis, horizons, E),axis=-1)
        inv_output = np.concatenate((scene,scene,scene),axis=-1)
        inv_pair = np.concatenate((inv_input,inv_output),axis=1)
        inv_pair = Image.fromarray( (255*inv_pair).astype('uint8'), mode='RGB')
        inv_pair.save(os.path.join(target, 'inversion', prefix + 'inv.png') ,'png')        

        # visibility: phi, x to psi
        vis_input = np.concatenate((scene, X, zero),axis=-1)
        vis_output = np.concatenate((vis,vis,vis),axis=-1)
        vis_pair = np.concatenate((vis_input,vis_output),axis=1)
        vis_pair = Image.fromarray( (255*vis_pair).astype('uint8'), mode='RGB')
        vis_pair.save(os.path.join(target, 'visibility', prefix + 'vis.png') ,'png')        

