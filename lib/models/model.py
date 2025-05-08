import torch
import torch.nn as nn
import copy
import numpy as np
from core.config import cfg
from models.module import Hand4Whole, ObjectRegressor
from models.transformer import TGformer
from models.templates import smplh, obj_dict
from funcs_utils import rot6d_to_aa, rot6d_to_rotmat, sample_joint_features, rigid_transform_3D
import torch.nn.functional as F

class HOITG(nn.Module):
    def __init__(self):
        super(HOITG, self).__init__()
        self.hand4whole = Hand4Whole()
        self.objregressor = ObjectRegressor()
        self.tgformer = TGformer()
        self.smplh_layer = copy.deepcopy(smplh.layer['neutral'])
        self.init_weights()
        self.trainable_modules = [self.hand4whole, self.objregressor,self.tgformer]

    def init_weights(self):
        checkpoint = torch.load('data/base_data/backbone_models/hand4whole.pth.tar')
        
        new_checkpoint = {}
        for k, v in checkpoint['network'].items():
            if k.startswith('module.'): new_checkpoint[k.replace('module.', '')] = v
        self.hand4whole.load_state_dict(new_checkpoint, strict=False)
        self.objregressor.init_weights()
        self.tgformer.init_weights()

    def load_weights(self, checkpoint):
        self.load_state_dict(checkpoint, strict=False)
    
    def upsample_features(self,features):
        upscaled = F.interpolate(features, size=(512, 512), mode='bilinear', align_corners=False)
        return upscaled.permute(0,2,3,1) 
    
    def forward(self, inputs):
        batch_size = inputs['img'].shape[0]

        # Initial reconstruction
        img_feat, hand_bbox, human_joint_img, human_pose, human_shape, cam_trans = self.hand4whole(inputs['img'], inputs['obj_id'])
        obj_pose, obj_trans = self.objregressor(img_feat)

        human_pose_ = rot6d_to_aa(human_pose.reshape(-1,6)).reshape(batch_size,-1)
        outputs = self.smplh_layer(betas=human_shape, global_orient=human_pose_[:, :3], body_pose=human_pose_[:, 3:66], left_hand_pose=human_pose_[:, 66:111], right_hand_pose=human_pose_[:, 111:])
        human_verts, human_joints = outputs.vertices - outputs.joints[:, [smplh.root_joint_idx]], outputs.joints - outputs.joints[:, [smplh.root_joint_idx]]
        human_verts = smplh.downsample(human_verts)
 
        obj_ids = inputs['obj_id'].reshape(-1).tolist()
        obj_pose_ = rot6d_to_rotmat(obj_pose).reshape(batch_size, 3, 3)
        obj_verts = torch.zeros((batch_size, 64, 3))
        for i, idx in enumerate(obj_ids):
            obj_verts[i] = torch.tensor(obj_dict[idx].load_verts())
        obj_verts = torch.matmul(torch.tensor(obj_verts).to(obj_pose_.device), obj_pose_.transpose(1,2)) + obj_trans[:,None]

        human_joints_proj = self.projection(human_joints, cam_trans)
        human_verts_proj = self.projection(human_verts, cam_trans)
        obj_verts_proj = self.projection(obj_verts, cam_trans)

        # base
        smpl_coord_xy = (human_verts_proj[...,:2].detach() + 0.5) * img_feat.shape[-1]
        H_tokens = sample_joint_features(img_feat, smpl_coord_xy) 
        obj_coord_xy = (obj_verts_proj[...,:2].detach() + 0.5) * img_feat.shape[-1]
        O_tokens = sample_joint_features(img_feat, obj_coord_xy)
        smpl_joint_xy = (human_joints_proj[...,:2].detach() + 0.5) * img_feat.shape[-1]
        joint_tokens = sample_joint_features(img_feat, smpl_joint_xy)


        humanjon_tokens, humanver_tokens, object_tokens = torch.cat((joint_tokens,human_joints.detach()),-1),torch.cat((H_tokens, human_verts.detach()), -1), torch.cat((O_tokens, obj_verts.detach()), -1)

        transformertokens = torch.cat((humanjon_tokens,humanver_tokens,object_tokens),dim=1)


        graph = inputs['objgraph'].cuda()
        # CRFormer
        refined_human_joints,refined_human_verts, refined_obj_verts = self.tgformer(transformertokens,graph)

        refined_human_joints_pro = self.projection(refined_human_joints, cam_trans)

        # Get final meshes 
        refined_obj_pose, refined_obj_trans = self.kps_to_Rt(refined_obj_verts, obj_ids)

        pred_vertices_17,pred_vertices_68 = smplh.getupsample(refined_human_verts)

        return {
            'initial_smpl_pose': human_pose,
            'initial_smpl_shape': human_shape,
            'initial_obj_pose': obj_pose, 
            'initial_obj_trans': obj_trans,

            'smpl_verts6000': pred_vertices_68,
            'smpl_verts1730': pred_vertices_17,
            'smpl_verts431': refined_human_verts,

            'obj_verts': refined_obj_verts,
            'obj_pose': refined_obj_pose,
            'obj_trans': refined_obj_trans,

            'h3d_keypoints': human_joints,
            'h2d_keypoints': human_joints_proj,
            'hp2d_keypoints': human_joint_img,

            'refh3d_keypoints': refined_human_joints,
            'refh2d_keypoints': refined_human_joints_pro,

            'hand_bbox': hand_bbox,
            'cam_trans': cam_trans,
        }

    def projection(self, coords, cam_trans):
        x = (coords[:,:,0] + cam_trans[:,None,0]) / (coords[:,:,2] + cam_trans[:,None,2] + 1e-4) * cfg.CAMERA.focal[0] / cfg.MODEL.input_body_shape[0]
        y = (coords[:,:,1] + cam_trans[:,None,1]) / (coords[:,:,2] + cam_trans[:,None,2] + 1e-4) * cfg.CAMERA.focal[1] / cfg.MODEL.input_body_shape[1]
        return torch.stack((x,y),2)

    def kps_to_Rt(self, pred_kps, obj_ids):
        template_kps = []
        for i in obj_ids:
            template_kps.append(torch.tensor(obj_dict[i].load_verts()))
        template_kps = torch.stack(template_kps).to(pred_kps.device)

        rot_list, trans_list = [], []
        for i in range(len(obj_ids)):
            scale, rot ,trans = rigid_transform_3D(template_kps[i].detach().cpu().numpy(), pred_kps[i].detach().cpu().numpy())
            rot_list.append(rot); trans_list.append(trans)
        rot_list = torch.tensor(rot_list, device=pred_kps.device)
        trans_list = torch.tensor(trans_list, device=pred_kps.device)

        return rot_list, trans_list


def get_model():
    return HOITG()
