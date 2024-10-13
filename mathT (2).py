import torch.nn as nn
import torch
import math
from math import *
import os
import pickle
import torch
import numpy as np
import cv2
from functools import partial



def block_diagonal_matrix_np(matrix2d_list):
    ret = np.zeros(sum([np.array(m.shape) for m in matrix2d_list]))
    r, c = 0, 0
    for m in matrix2d_list:
        lr, lc = m.shape
        ret[r:r+lr, c:c+lc] = m
        r += lr
        c += lc
    return ret


def lerp(a, b, t):
    return a * (1 - t) + b * t

def append_value(x: torch.Tensor, value: float, dim=-1):
    app = torch.ones_like(x.index_select(dim, torch.tensor([0], device=x.device))) * value
    x = torch.cat((x, app), dim=dim)
    return x
append_zero = partial(append_value, value=0)
append_one = partial(append_value, value=1)
def axis_angle_to_rotation_matrix(a: torch.Tensor):

    axis, angle = normalize_tensor(a.view(-1, 3), return_norm=True)
    axis[torch.isnan(axis)] = 0
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    c, s = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    r = c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * vector_cross_matrix(axis)
    return r

def axis_angle_to_rotation_matrix_T(a: torch.Tensor):

    axis, angle = normalize_tensor(a.view(-1, 3), return_norm=True)
    axis[torch.isnan(axis)] = 0
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    c, s = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    r = c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * vector_cross_matrix(axis)
    return r
def vector_cross_matrix(x: torch.Tensor):

    x = x.view(-1, 3)
    zeros = torch.zeros(x.shape[0], device=x.device)
    return torch.stack((zeros, -x[:, 2], x[:, 1],
                        x[:, 2], zeros, -x[:, 0],
                        -x[:, 1], x[:, 0], zeros), dim=1).view(-1, 3, 3)

def vector_cross_matrix_np(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]], dtype=float)


def normalize_angle(q):

    mod = q % (2 * np.pi)
    mod[mod >= np.pi] -= 2 * np.pi
    return mod

def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False):

    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / norm
    return normalized_x if not return_norm else (normalized_x, norm)


def axis_angle_to_rotation_matrix(a: torch.Tensor):

    axis, angle = normalize_tensor(a.view(-1, 3), return_norm=True)
    axis[torch.isnan(axis)] = 0
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    c, s = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    r = c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * vector_cross_matrix(axis)
    return r.view(a.shape[0],-1,3,3)





def transformation_matrix(R: torch.Tensor, p: torch.Tensor):

    Rp = torch.cat((R, p.unsqueeze(-1)), dim=-1)
    OI = torch.cat((torch.zeros(list(Rp.shape[:-2]) + [1, 3], device=R.device),
                    torch.ones(list(Rp.shape[:-2]) + [1, 1], device=R.device)), dim=-1)
    T = torch.cat((Rp, OI), dim=-2)
    return T


def decode_transformation_matrix(T: torch.Tensor):

    R = T[..., :3, :3].clone()
    p = T[..., :3, 3].clone()
    return R, p


def inverse_transformation_matrix(T: torch.Tensor):

    R, p = decode_transformation_matrix(T)
    invR = R.transpose(-1, -2)
    invp = -torch.matmul(invR, p.unsqueeze(-1)).squeeze(-1)
    invT = transformation_matrix(invR, invp)
    return invT


def _forward_tree(x_local: torch.Tensor, parent, reduction_fn):
    x_global = [x_local[:, 0]]
    for i in range(1, len(parent)):
        x_global.append(reduction_fn(x_global[parent[i]], x_local[:, i]))
    x_global = torch.stack(x_global, dim=1)
    return x_global
def rotation_matrix_to_axis_angle(r: torch.Tensor):
    result = [cv2.Rodrigues(_)[0] for _ in r.clone().detach().cpu().view(-1, 3, 3).numpy()]
    result = torch.from_numpy(np.stack(result)).float().squeeze(-1).to(r.device)
    return result
def bone_vector_to_joint_position(bone_vec: torch.Tensor, parent):

    bone_vec = bone_vec.view(bone_vec.shape[0], -1, 3)
    joint_pos = _forward_tree(bone_vec, parent, torch.add)
    return joint_pos
def joint_position_to_bone_vector(joint_pos: torch.Tensor, parent):
    joint_pos = joint_pos.view(joint_pos.shape[0], -1, 3)
    bone_vec = _inverse_tree(joint_pos, parent, torch.add, torch.neg)
    return bone_vec

def forward_kinematics_R(R_local: torch.Tensor, parent):

    R_local = R_local.view(R_local.shape[0], -1, 3, 3)
    R_global = _forward_tree(R_local, parent, torch.bmm)
    return R_global


def inverse_kinematics_R(R_global: torch.Tensor, parent):

    R_global = R_global.view(R_global.shape[0], -1, 3, 3)
    R_local = _inverse_tree(R_global, parent, torch.bmm, partial(torch.transpose, dim0=1, dim1=2))
    return R_local


def forward_kinematics_T(T_local: torch.Tensor, parent):

    T_local = T_local.view(T_local.shape[0], -1, 4, 4)
    T_global = _forward_tree(T_local, parent, torch.bmm)
    return T_global


def inverse_kinematics_T(T_global: torch.Tensor, parent):

    T_global = T_global.view(T_global.shape[0], -1, 4, 4)
    T_local = _inverse_tree(T_global, parent, torch.bmm, inverse_transformation_matrix)
    return T_local


def forward_kinematics(R_local: torch.Tensor, p_local: torch.Tensor, parent):

    R_local = R_local.view(R_local.shape[0], -1, 3, 3)
    p_local = p_local.view(p_local.shape[0], -1, 3)
    T_local = transformation_matrix(R_local, p_local)
    T_global = forward_kinematics_T(T_local, parent)
    return decode_transformation_matrix(T_global)
class ParametricModel:
    def __init__(self, official_model_file="/root/autodl-tmp/SMPL_NEUTRAL.pkl", use_pose_blendshape=False, device=torch.device('cuda')):

        with open(official_model_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        self._J_regressor = torch.from_numpy(data['J_regressor'].toarray()).float().to(device)
        self._skinning_weights = torch.from_numpy(data['weights']).float().to(device)
        self._posedirs = torch.from_numpy(data['posedirs']).float().to(device)
        self._shapedirs = torch.from_numpy(np.array(data['shapedirs'])).float().to(device)
        self._v_template = torch.from_numpy(data['v_template']).float().to(device)
        self._J = torch.from_numpy(data['J']).float().to(device)
        self.face = data['f']
        self.parent = data['kintree_table'][0].tolist()
        self.parent[0] = None
        self.use_pose_blendshape = use_pose_blendshape

    def save_obj_mesh(self, vertex_position, file_name='a.obj'):

        with open(file_name, 'w') as fp:
            for v in vertex_position:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.face + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    @staticmethod
    def save_unity_motion(pose: torch.Tensor = None, tran: torch.Tensor = None, output_dir='saved_motions/'):

        os.makedirs(output_dir, exist_ok=True)

        if pose is not None:
            f = open(os.path.join(output_dir, 'pose.txt'), 'w')
            pose = rotation_matrix_to_axis_angle(pose).view(pose.shape[0], -1)
            f.write('\n'.join([','.join(['%.4f' % _ for _ in p]) for p in pose]))
            f.close()

        if tran is not None:
            f = open(os.path.join(output_dir, 'tran.txt'), 'w')
            f.write('\n'.join([','.join(['%.5f' % _ for _ in t]) for t in tran.view(tran.shape[0], 3)]))
            f.close()

    def get_zero_pose_joint_and_vertex(self, shape: torch.Tensor = None):

        if shape is None:
            j, v = self._J - self._J[:1], self._v_template - self._J[:1]
        else:
            shape = shape.view(-1, 10)
            v = torch.tensordot(shape, self._shapedirs, dims=([1], [2])) + self._v_template
            j = torch.matmul(self._J_regressor, v)
            j, v = j - j[:, :1], v - j[:, :1]
        return j, v

    def bone_vector_to_joint_position(self, bone_vec: torch.Tensor):

        return bone_vector_to_joint_position(bone_vec, self.parent)

    def joint_position_to_bone_vector(self, joint_pos: torch.Tensor):

        return joint_position_to_bone_vector(joint_pos, self.parent)

    def forward_kinematics_R(self, R_local: torch.Tensor):

        return forward_kinematics_R(R_local, self.parent)

    def inverse_kinematics_R(self, R_global: torch.Tensor):

        return inverse_kinematics_R(R_global, self.parent)

    def forward_kinematics_T(self, T_local: torch.Tensor):

        return forward_kinematics_T(T_local, self.parent)

    def inverse_kinematics_T(self, T_global: torch.Tensor):

        return inverse_kinematics_T(T_global, self.parent)

    def forward_kinematics(self, pose: torch.Tensor, shape: torch.Tensor = None, tran: torch.Tensor = None,
                           calc_mesh=False):

        def add_tran(x):
            return x if tran is None else x + tran.view(-1, 1, 3)

        # pose = pose.view(pose.shape[0], -1, 3, 3)
        pose=pose.view(pose.shape[0],-1,3)
        pose=axis_angle_to_rotation_matrix(pose)
        
        j, v = [_.expand(pose.shape[0], -1, -1) for _ in self.get_zero_pose_joint_and_vertex(shape)]
        T_local = transformation_matrix(pose, self.joint_position_to_bone_vector(j))
        T_global = self.forward_kinematics_T(T_local)
        pose_global, joint_global = decode_transformation_matrix(T_global)
        if calc_mesh is False:
            return pose_global, add_tran(joint_global)

        T_global[..., -1:] -= torch.matmul(T_global, append_zero(j, dim=-1).unsqueeze(-1))
        T_vertex = torch.tensordot(T_global, self._skinning_weights, dims=([1], [1])).permute(0, 3, 1, 2)
        if self.use_pose_blendshape:
            r = (pose[:, 1:] - torch.eye(3, device=pose.device)).flatten(1)
            v = v + torch.tensordot(r, self._posedirs, dims=([1], [2]))
        vertex_global = torch.matmul(T_vertex, append_one(v, dim=-1).unsqueeze(-1)).squeeze(-1)[..., :3]
        return pose_global, add_tran(joint_global), add_tran(vertex_global)

    def view_joint(self, joint_list: list, fps=60, distance_between_subjects=0.8):

        import vctoolkit as vc
        import vctoolkit.viso3d as vo3d
        joint_list = [(j.view(-1, len(self.parent), 3) - j.view(-1, len(self.parent), 3)[:1, :1]).cpu().numpy()
                      for j in joint_list]

        v_list, f_list = [], []
        f = vc.joints_to_mesh(joint_list[0][0], self.parent)[1]
        for i in range(len(joint_list)):
            v = np.stack([vc.joints_to_mesh(frame, self.parent)[0] for frame in joint_list[i]])
            v[:, :, 0] += distance_between_subjects * i
            v_list.append(v)
            f_list.append(f.copy())
            f += v.shape[1]

        verts = np.concatenate(v_list, axis=1)
        faces = np.concatenate(f_list)
        if verts.shape[0] > 1:
            vo3d.render_sequence_3d(verts, faces, 720, 720, 'a.mp4', fps, visible=True)
        else:
            vo3d.vis_mesh(verts[0], faces)

    def view_mesh(self, vertex_list: list, fps=60, distance_between_subjects=0.8):

        import vctoolkit.viso3d as vo3d
        v_list, f_list = [], []
        f = self.face.copy()
        for i in range(len(vertex_list)):
            v = vertex_list[i].clone().view(-1, self._v_template.shape[0], 3)
            v[:, :, 0] += distance_between_subjects * i
            v_list.append(v)
            f_list.append(f.copy())
            f += v.shape[1]

        verts = torch.cat(v_list, dim=1).cpu().numpy()
        faces = np.concatenate(f_list)
        if verts.shape[0] > 1:
            vo3d.render_sequence_3d(verts, faces, 720, 720, 'a.mp4', fps, visible=True)
        else:
            vo3d.vis_mesh(verts[0], faces)

    def view_motion(self, pose_list: list, tran_list: list = None, fps=60, distance_between_subjects=0.8):

        verts = []
        for i in range(len(pose_list)):
            pose = pose_list[i].view(-1, len(self.parent), 3, 3)
            tran = tran_list[i].view(-1, 3) - tran_list[i].view(-1, 3)[:1] if tran_list else None
            verts.append(self.forward_kinematics(pose, tran=tran, calc_mesh=True)[2])
        self.view_mesh(verts, fps, distance_between_subjects=distance_between_subjects)
        
class joint_set:
    leaf = [18,19, 20, 21]
    full = list(range(1, 24))
    reduced = [0,1,2,3,4,5,6,9,12,13,14,15,16,17,18,19,]
    ignored = [7,8,10,11,20,21,22,23]
    n_reduced=len(reduced)
def cal_pos_mse(out_rot, y_pos):
    glb_reduced_pose = out_rot.to("cuda").view(-1, joint_set.n_reduced, 3)
    global_full_pose = torch.zeros(3).repeat(glb_reduced_pose.shape[0], 24, 1).to("cuda")
    global_full_pose[:, joint_set.reduced] = glb_reduced_pose
    global_full_pose[:, joint_set.ignored] = torch.zeros(3, device="cuda")

    pos_test= y_pos.to("cuda").view(-1, joint_set.n_reduced, 3)
    pos_full=torch.zeros(3).repeat(pos_test.shape[0], 24, 1).to("cuda")
    pos_full[:, joint_set.reduced] = pos_test.view(-1,16,3).to("cuda")
    pos_full[:, joint_set.ignored] = torch.zeros(3, device="cuda")
    body_model=ParametricModel()
    rori,pos=body_model.forward_kinematics(global_full_pose.view(-1,72).to("cuda"))
    loss_fn=torch.nn.MSELoss()
    return loss_fn(pos,pos_full)

def cal_jittor_mse(out_rot, y_jittor):
    glb_reduced_pose = out_rot.to("cuda").view(-1, joint_set.n_reduced, 3)
    global_full_pose = torch.zeros(3).repeat(glb_reduced_pose.shape[0], 24, 1).to("cuda")
    global_full_pose[:, joint_set.reduced] = glb_reduced_pose
    global_full_pose[:, joint_set.ignored] = torch.zeros(3, device="cuda")

    # pos_test= y_pos.to("cuda").view(-1, joint_set.n_reduced, 3)
    # pos_full=torch.zeros(3).repeat(pos_test.shape[0], 24, 1).to("cuda")
    # pos_full[:, joint_set.reduced] = pos_test.view(-1,16,3).to("cuda")
    # pos_full[:, joint_set.ignored] = torch.zeros(3, device="cuda")
    body_model=ParametricModel()
    rori,pos=body_model.forward_kinematics(global_full_pose.view(-1,72).to("cuda"))
    jittor=caljitter(pos)
    jittor=jittor[:,joint_set.reduced]
    loss_fn=torch.nn.MSELoss()
    return loss_fn(jittor.flatten(1),y_jittor[2:])
    # return jittor,y_jittor
def fk_solver(r,full_pose=True):
    glb_reduced_pose = r.to("cuda").view(-1, joint_set.n_reduced, 3)
    global_full_pose = torch.zeros(3).repeat(glb_reduced_pose.shape[0], 24, 1).to("cuda")
    global_full_pose[:, joint_set.reduced] = glb_reduced_pose
    global_full_pose[:, joint_set.ignored] = torch.zeros(3, device="cuda")
    return global_full_pose