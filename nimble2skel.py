"""
Run a forward pass of the SKEL model with default parameters (T pose) and export the resulting meshes.
Author: Marilyn Keller
"""

import os
import sys
import torch
import numpy as np 
from skel.skel_model import SKEL
from skel.kin_skel import pose_param_names
import trimesh

import polyscope as ps 
import polyscope.imgui as psim
from mot_loader import read_header,storage_to_numpy


def mot2skel(mot_data,osim_headers):
    mot_data = mot_data.astype(np.float32)

    header2ind = {k:i for i,k in enumerate(osim_headers)}
    
    T = mot_data.shape[0]
    D = len(pose_param_names)

    pose_params = np.zeros((T,D),dtype=np.float32)

    for d in range(D):
        skel_param_name = pose_param_names[d]
        if skel_param_name in header2ind:

            pose_params[:,d] = mot_data[:,header2ind[skel_param_name]]

    trans = mot_data[:,[header2ind[k] for k in ['pelvis_tx','pelvis_ty','pelvis_tz']]]    

    pose_params = np.pi*pose_params/180 # Convert to radians from degrees

    return pose_params,trans

if __name__ == '__main__':

    debug = True
    device = 'cuda'
    gender = 'male'


    if len(sys.argv) < 2:
        print('Usage: python quickstart.py <path_to_mocap_file>')
        mot_file = "/media/shubh/Elements/RoseYu/UCSD-OpenCap-Fitness-Dataset/MCS_DATA/OpenSim/OpenCapData_000cffd9-e154-4ce5-a075-1b4e1fd66201/OpenSimData/Pred_Kinematics/bap01.mot"
    else: 
        mot_file = sys.argv[1]




    osim_headers = read_header(mot_file)
    mot_data = storage_to_numpy(mot_file)


    if debug: 
        mot_data = mot_data[:100]


    pose_params,trans = mot2skel(mot_data,osim_headers)
    

    
    skel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'body_models','skel')
    print(skel_path)
    skel = SKEL(gender='female',model_path=skel_path).to(device)

    # Inittialize parameters with parameters to mot values (T pose)
    pose = torch.from_numpy(pose_params).to(device) # (T, 46)
    betas = torch.zeros(pose.shape[0], skel.num_betas).to(device) # (1, 10)
    trans = torch.from_numpy(trans).to(device) # (120, 3).to(device)

    # SKEL forward pass
    skel_output = skel(pose, betas, trans)

    output = {}
    output['joints_points'] = skel_output.joints.detach().cpu().numpy()
    output['joints_ori'] = skel_output.joints_ori.detach().cpu().numpy()
    output['skeleton_verts'] = skel_output.skel_verts.detach().cpu().numpy()
    output['skin_verts'] = skel_output.skin_verts.detach().cpu().numpy()



    # Init polyscope
    ps.init()
    
    ps_data = {}
    ps_data['skeleton_verts'] = ps.register_surface_mesh("Skeleton", output['skeleton_verts'][0],skel.skel_f.detach().cpu().numpy())
    ps_data['joints_points'] = ps.register_point_cloud("Joints", output['joints_points'][0])

    # Create callback to update the animation
    ps_scene = {'t': 0, 'is_paused': False, 'ui_text': "Enter instructions here",
                'T':output['skeleton_verts'].shape[0]}
    def callback():
        
        ########### Checks ############
        # Ensure self.t lies between 
        ps_scene['t'] %= ps_scene['T']

        for obj in ps_data:
            name, obj_type = obj.split('_')
            if obj_type == 'verts':
                ps_data[obj].update_vertex_positions(output[obj][ps_scene['t']])
            elif obj_type == 'points':
                ps_data[obj].update_point_positions(output[obj][ps_scene['t']])

        if not ps_scene['is_paused']: 
            ps_scene['t'] += 1 


        # Keyboard control - to toggle pause  for spacebar press
        if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Space)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Space)):            
            ps_scene['is_paused'] = not ps_scene['is_paused']

        # Left arrow pressed
        if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_LeftArrow)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_LeftArrow)):
            ps_scene['t'] -= 1

        if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_RightArrow)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_RightArrow)):
            ps_scene['t'] += 1


        ############## GUI to  update the animations ##################
        # Input text
        changed, ps_scene["ui_text"] = psim.InputText("- Coach Instructions", ps_scene["ui_text"])


        # Create a floater to show the timestep and adject self.t accordingly
        changed, ps_scene['t'] = psim.SliderInt("", ps_scene['t'], v_min=0, v_max=ps_scene['T'])
        psim.SameLine()

        # Create a render button which when pressed will create a .mp4 file
        if psim.Button("<"):
            ps_scene['t'] -= 1
        
        psim.SameLine()
        if psim.Button("Play Video" if ps_scene['is_paused'] else "Pause Video"):
            ps_scene['is_paused'] = not ps_scene['is_paused']

        psim.SameLine()
        if psim.Button(">"):
            ps_scene['t'] += 1

        # psim.SameLine()
       # if psim.Button("Render Video"):
        #     self.render_video()        `

        # if(psim.TreeNode("Load Experiment")):

        #     # psim.TextUnformatted("Load Optimized samples")

        #     changed = psim.BeginCombo("- Experiement", ps_scene["experiment_options_selected"])
        #     if changed:
        #         for val in ps_scene["experiment_options"]:
        #             _, selected = psim.Selectable(val, selected=ps_scene["experiment_options_selected"]==val)
        #             if selected:
        #                 ps_scene["experiment_options_selected"] = val
        #         psim.EndCombo()

        #     changed = psim.BeginCombo("- Category", ps_scene["category_options_selected"])
        #     if changed:
        #         for val in ps_scene["category_options"]:
        #             _, selected = psim.Selectable(val, selected=ps_scene["category_options_selected"]==val)
        #             if selected:
        #                 ps_scene["category_options_selected"] = val
        #         psim.EndCombo()


    ps.set_user_callback(callback)
    ps.show()

    # # Export meshes    
    # os.makedirs('output', exist_ok=True)
    # skin_mesh_path = os.path.join('output', f'skin_mesh_{gender}.obj')
    # skeleton_mesh_path = os.path.join('output', f'skeleton_mesh_{gender}.obj')
    
    # trimesh.Trimesh(vertices=output['skin_verts'].detach().cpu().numpy()[0], 
    #                 faces=skel.skin_f.cpu()).export(skin_mesh_path)
    # print('Skin mesh saved to: {}'.format(skin_mesh_path))
    
    # trimesh.Trimesh(vertices=output['skel_verts'].detach().cpu().numpy()[0],
    #                 faces=skel.skel_f.cpu()).export(skeleton_mesh_path)
    # print('Skeleton mesh saved to: {}'.format(skeleton_mesh_path))