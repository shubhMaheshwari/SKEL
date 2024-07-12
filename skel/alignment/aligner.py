
"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Soyong Shin, Marilyn Keller
See https://skel.is.tue.mpg.de/license.html for licensing and contact information.
"""

import argparse
import math
import os
import pickle
from skel.alignment.losses import compute_pose_loss, compute_scapula_loss, compute_spine_loss, compute_time_loss, pretty_loss_print
from skel.alignment.utils import to_numpy, to_params, to_torch
import torch
import numpy as np
from tqdm import trange
import smplx
import torch.nn.functional as F
from psbody.mesh import Mesh, MeshViewer, MeshViewers
import skel.config as cg
from skel.skel_model import SKEL


class SkelFitter(object):
    
    def __init__(self, gender, device, num_betas=10, export_meshes=False) -> None:

        self.smpl = smplx.create(cg.smpl_folder, model_type='smpl', gender=gender, num_betas=num_betas, batch_size=1, export_meshes=False).to(device)
        self.skel = SKEL(gender).to(device)
        self.gender = gender
        self.device = device
        self.num_betas = num_betas
        
        # Instanciate masks used for the vertex to vertex fitting
        fitting_mask_file = 'skel/alignment/riggid_parts_mask.pkl'
        fitting_indices = pickle.load(open(fitting_mask_file, 'rb'))
        fitting_mask = torch.zeros(6890, dtype=torch.float)
        fitting_mask[fitting_indices] = 1
        self.fitting_mask = fitting_mask.reshape(1, -1, 1).to(self.device) # 1xVx1 to be applied to verts that are BxVx3
        
        smpl_torso_joints = [0,3,6,9]
        verts_mask = (self.smpl.lbs_weights[:,smpl_torso_joints]>0.5).sum(dim=-1)>0
        self.torso_verts_mask = verts_mask.unsqueeze(0).unsqueeze(-1) # Because verts are of shape BxVx3
        
        self.export_meshes = export_meshes
        
           
        self.cfg = {
            'lr': 1,
            'max_iter': 25,
            'num_steps': 10,
            'line_search_fn': 'None', #'strong_wolfe',
            'rot_only' : True, # Only optimize the global rotation


            'l_verts': 0.1,
            'l_verts_loose': 0.0,         
            'l_time_loss': 1e-2,      
            'l_joint': 0.0,
            'l_scapula_loss': 0.0,
            'l_spine_loss': 0.0,
            'l_pose_loss': 0.0,
           

            'pose_reg_factor': 1e1

        }
        
        self.cfg_optim = {
            'lr': 1e-1,
            'max_iter': 10,
            'num_steps': 10,
            'rot_only': False,
            
            'l_verts_loose': 5e2,         
            'l_joint': 1e2,
            'l_scapula_loss': 1e-2,
            'l_spine_loss': 1e-3,
            'l_pose_loss': 1e-4,
           

            'pose_reg_factor': 1e1
        }


        # make the cfg being an object using omegaconf
        import omegaconf    
        self.cfg =  omegaconf.OmegaConf.create(self.cfg)
        
        
    def run_fit(self, 
            trans_in, 
            betas_in, 
            poses_in, 
            batch_size=20, 
            skel_data_init=None, 
            force_recompute=False, 
            debug=False,
            watch_frame=0,
            freevert_mesh=None):
        """Align SKEL to a SMPL sequence."""

        self.nb_frames = poses_in.shape[0]
        self.watch_frame = watch_frame
        
        print('Fitting {} frames'.format(self.nb_frames))
        
        to_params = lambda x: torch.from_numpy(x).float().to(self.device).requires_grad_(True)
        to_torch = lambda x: torch.from_numpy(x).float().to(self.device)
        
        if skel_data_init is None or force_recompute:
        
            poses_skel = np.zeros((self.nb_frames, self.skel.num_q_params))
            poses_skel[:, :3] = poses_in[:, :3].copy() # Global orient are similar between SMPL and SKEL, so init with SMPL angles
            
            betas_skel = np.zeros((self.nb_frames, 10)); 
            betas_skel[:] = betas_in[..., :10].copy()
            # betas_out = smpl_data.betas[..., :10].reshape(-1, 10).expand(nb_frames, -1).detach().cpu().numpy()
            
            trans_skel = trans_in.copy() # Translation is similar between SMPL and SKEL, so init with SMPL translation
            
        else:
            # Load from previous alignment
            poses_skel = skel_data_init['poses']
            betas_skel = skel_data_init['betas']
            trans_skel = skel_data_init['trans']
        
        # We cut the whole sequence in batches for parallel optimization  
        if batch_size > self.nb_frames:
            batch_size = self.nb_frames
            print('Batch size is larger than the number of frames. Setting batch size to {}'.format(batch_size))
            
        n_batch = math.ceil(self.nb_frames/batch_size)
        pbar = trange(n_batch, desc='Running batch optimization')
        
        # initialize the res dict to store the per frame result skel parameters
        out_keys = ['poses', 'betas', 'trans'] 
        if self.export_meshes:
            out_keys += ['skel_v', 'skel_f', 'skin_v', 'skin_f', 'smpl_v', 'smpl_f']
        res_dict = {key: [] for key in out_keys}
     
        for i in pbar:
                  
            if debug:
                # Only run the first batch to test, ignore the rest
                if i > 1:
                    continue
            
            # Get mini batch
            i_start =  i * batch_size
            i_end = min((i+1) * batch_size, self.nb_frames)
            
            # self.fit_batch()
            # SMPL params
            poses_smpl = to_torch(poses_in[i_start:i_end].copy())
            betas_smpl = to_torch(betas_in[:self.num_betas].copy()).expand(i_end-i_start, -1)
            trans_smpl = to_torch(trans_in[i_start:i_end].copy())
            
            # Run a SMPL forward pass to get the SMPL body vertices
            smpl_output = self.smpl(betas=betas_smpl, body_pose=poses_smpl[:,3:], transl=trans_smpl, global_orient=poses_smpl[:,:3])
            verts = smpl_output.vertices
            if(freevert_mesh is not None):
                verts = to_torch(freevert_mesh).unsqueeze(0).repeat_interleave(batch_size, 0)
            
            # SKEL params        
            poses = to_params(poses_skel[i_start:i_end].copy())
            betas = to_params(betas_skel[i_start:i_end].copy())
            trans = to_params(trans_skel[i_start:i_end].copy())
            
            if i == 0 and not skel_data_init:
                # Optimize the global rotation and translation for the initial fitting
                cfg = self.cfg
                self.optim([trans,poses], poses, betas, trans, verts, cfg)
                cfg.update(self.cfg_optim)
                self.optim([trans,poses], poses, betas, trans, verts, cfg)
            else:
                cfg = self.cfg
                cfg.update(self.cfg_optim)
                self.optim([trans,poses], poses, betas, trans, verts, cfg)
                
            # Save the result
            poses_skel[i_start:i_end] = poses[:].detach().cpu().numpy().copy()
            trans_skel[i_start:i_end] = trans[:].detach().cpu().numpy().copy()
            
            # Initialize the next frames with current frame
            poses_skel[i_end:] = poses[-1:].detach().cpu().numpy().copy()
            trans_skel[i_end:] = trans[-1].detach().cpu().numpy().copy()
            betas_skel[i_end:] = betas[-1:].detach().cpu().numpy().copy()

            res_dict['poses'].append(poses.detach().cpu().numpy().copy())
            res_dict['betas'].append(betas.detach().cpu().numpy().copy())
            res_dict['trans'].append(trans.detach().cpu().numpy().copy())
            res_dict['gender'] = self.gender
            
            if self.export_meshes:
                # Export the meshes
                skel_output = self.skel.forward(poses=poses, betas=betas, trans=trans, poses_type='skel', skelmesh=True)
                res_dict['skel_v'].append(skel_output.skel_verts.detach().cpu().numpy().copy())
                res_dict['skin_v'].append(skel_output.skin_verts.detach().cpu().numpy().copy())
                res_dict['smpl_v'].append(verts.detach().cpu().numpy().copy())
                res_dict['skel_f'] = self.skel.skel_f.cpu().numpy().copy()
                res_dict['skin_f'] = self.skel.skin_f.cpu().numpy().copy()
                res_dict['smpl_f'] = self.smpl.faces
            
        for key, val in res_dict.items():
            if isinstance(val, list):
                res_dict[key] = np.concatenate(val)
                
        return res_dict
    
    def optim(self,
            params, 
            poses,
            betas,
            trans,
            verts,
            cfg,
            ):
        
            # poseLoss = PoseLimitLoss().to(device)
            
            # regress joints 
            anat_joints = torch.einsum('bik,ji->bjk', [verts, self.skel.J_regressor_osim]) 
            dJ=torch.zeros((poses.shape[0], 24, 3), device=betas.device)
            

            # optimizer = torch.optim.LBFGS(params, lr=cfg.lr, max_iter=cfg.max_iter, line_search_fn=cfg.line_search_fn)
            optimizer = torch.optim.LBFGS(params, lr=cfg.lr, max_iter=cfg.max_iter, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100)

            pbar = trange(cfg.num_steps, leave=False)
            if('DISABLE_VIEWER' in os.environ):
                mv = None
                print("\n DISABLE_VIEWER flag is set, running in headless mode")
            else:
                mv = MeshViewers((1,2),  keepalive=False)

            def closure():
                optimizer.zero_grad()
                
                fi = self.watch_frame #frame of the batch to display
                output = self.skel.forward(poses=poses[fi:fi+1], 
                                            betas=betas[fi:fi+1], 
                                            trans=trans[fi:fi+1], 
                                            poses_type='skel', 
                                            dJ=dJ[fi:fi+1],
                                            skelmesh=True)
                    
                def fstep_plot(output, mv, fi):
                    if cfg.rot_only:
                        mask = self.torso_verts_mask
                    else:
                        mask = self.fitting_mask 
                        
                    skin_err_value = ((output.skin_verts[fi] - verts[fi])**2).sum(dim=-1).sqrt()
                    skin_err_value = skin_err_value / 0.05
                    skin_err_value = to_numpy(skin_err_value)
                        
                    skin_mesh = Mesh(v=to_numpy(output.skin_verts[fi]), f=[], vc='white')
                    skel_mesh = Mesh(v=to_numpy(output.skel_verts[fi]), f=self.skel.skel_f.cpu().numpy(), vc='white')
                    
                    # Display vertex distance on SMPL
                    smpl_verts = to_numpy(verts[fi])
                    smpl_mesh = Mesh(v=smpl_verts, f=self.smpl.faces)
                    smpl_mesh.set_vertex_colors_from_weights(skin_err_value, scale_to_range_1=False)       
                    
                    smpl_mesh_masked = Mesh(v=smpl_verts[to_numpy(mask[0,:,0])], f=[], vc='green')
                    
                    # List the meshes to display
                    meshes_left = [smpl_mesh, skel_mesh]
                    meshes_right = [smpl_mesh_masked, skin_mesh, skel_mesh]
                
                   
                    # + location_to_spheres(output.joints.detach().cpu().numpy()[frame_to_watch], color=(1,0,0), radius=0.02)
                    # + [Mesh(v=verts[frame_to_watch].detach().cpu().numpy(), f=smpl_data.faces, vc='green')] \
                    # + location_to_spheres(joints_reg[frame_to_watch].detach().cpu().numpy(), color=(0,1,0), radius=0.02) \
                    if('DISABLE_VIEWER' not in os.environ):
                        mv[0][0].set_dynamic_meshes(meshes_left)
                        mv[0][1].set_dynamic_meshes(meshes_right)
                
                    # print(poses[frame_to_watch, :3])
                    # print(trans[frame_to_watch])
                    # print(betas[frame_to_watch, :3])
                    # mv.get_keypress()
            
                fstep_plot(output, mv, fi)
                    
                loss = self.fitting_loss(poses,
                                        betas,
                                        trans,
                                        dJ,
                                        anat_joints,
                                        verts,
                                        cfg)
                                        
                loss.backward()
            
                return loss


            for step_i in pbar:
                loss = optimizer.step(closure).item()
                # with torch.no_grad():
                #     poses[:] = torch.atan2(poses.sin(), poses.cos())
                pbar.set_postfix_str(f"Loss {loss:.4f}")
            
    def fitting_loss(self,
                    poses,
                    betas,
                    trans,
                    dJ,
                    anat_joints,
                    verts,
                    cfg):
        
        loss_dict = {}
        
        if cfg.rot_only:
            # Only optimize the global rotation of the body, i.e. the first 3 angles of the pose
            joint_mask = torch.zeros_like(poses).to(self.device)
            joint_mask[:,:3] = 1
            poses_in = poses * joint_mask
            
            # Only fit the thorax vertices to recover the proper body orientation and translation
            verts_mask = self.torso_verts_mask
        else:
            poses_in = poses
            verts_mask = torch.ones_like(self.fitting_mask )
    
        output = self.skel.forward(poses=poses_in, betas=betas, trans=trans, poses_type='skel', dJ=dJ, skelmesh=False)
        
        # Fit the SMPL vertices
        # We know the skinning of the forearm and the neck are not perfect,
        # so we create a mask of the SMPL vertices that are important to fit, like the hands and the head
        loss_dict['verts_loss'] = cfg.l_verts * (verts_mask * self.fitting_mask * (output.skin_verts - verts)**2).sum() / (self.fitting_mask*verts_mask).sum()
        # loss_dict['verts_loss'] = cfg.l_verts * F.mse_loss(output.skin_verts, verts)
        loss_dict['verts_loss_loose'] = cfg.l_verts_loose * F.mse_loss(output.skin_verts, verts)
        
        # Fit the regressed joints
        loss_dict['joint_loss'] = cfg.l_joint * F.mse_loss(output.joints, anat_joints)
        
        # Time consistancy
        if poses.shape[0] > 1:
            loss_dict['time_loss'] = cfg.l_time_loss * F.mse_loss(poses[1:], poses[:-1])
            
        # Regularize the pose
        loss_dict['scapula_loss'] = cfg.l_scapula_loss * compute_scapula_loss(poses_in)
        loss_dict['spine_loss'] = cfg.l_spine_loss * compute_spine_loss(poses_in)
        loss_dict['pose_loss'] = cfg.l_pose_loss * compute_pose_loss(poses)
        
        # Adjust the losses of all the pose regularizations sub losses with the pose_reg_factor value
        for key in ['scapula_loss', 'spine_loss', 'pose_loss']:
            loss_dict[key] = cfg.pose_reg_factor * loss_dict[key]
        
        loss = sum(loss_dict.values())
        print(pretty_loss_print(loss, loss_dict))
                
        return loss

        