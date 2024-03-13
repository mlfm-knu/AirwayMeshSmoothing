import numpy as np
import torch
import copy
from typing import Union
from codes.compute_mesh import Mesh


def vertex_updating(pos: torch.Tensor, norm: torch.Tensor, mesh: Mesh, loop=10) -> torch.Tensor:
    new_pos = pos.detach().clone()
    norm = norm.detach().clone()
    for iter in range(loop):
        fc = torch.sum(new_pos[mesh.faces], 1) / 3.0
        for i in range(len(new_pos)):
            cis = fc[list(mesh.vf[i])]
            nis = norm[list(mesh.vf[i])]
            cvis = cis - new_pos[i].reshape(1, -1)
            ncvis = torch.sum(nis * cvis, dim=1)
            dvi = torch.sum(ncvis.reshape(-1, 1) * nis, dim=0)
            dvi /= len(mesh.vf[i])
            new_pos[i] += dvi
    return new_pos

def ver_loss(pred_pos: Union[torch.Tensor, np.ndarray], real_pos: np.ndarray, ltype="RMSE") -> torch.Tensor:
    # loss for vertex positions
    if type(pred_pos) == np.ndarray:
        pred_pos = torch.from_numpy(pred_pos)
    real_pos = torch.from_numpy(real_pos).to(pred_pos.device)

    if ltype == "MAE":
        diff_pos = torch.sum(torch.abs(real_pos - pred_pos), dim=1)
        loss = torch.sum(diff_pos) / len(diff_pos)

    elif ltype == "RMSE":
        diff_pos = torch.abs(real_pos - pred_pos)
        diff_pos = diff_pos ** 2
        diff_pos = torch.sum(diff_pos.squeeze(), dim=1)
        diff_pos = torch.sum(diff_pos) / len(diff_pos)
        loss = torch.sqrt(diff_pos + 1.0e-6)
    else:
        exit()
    return loss

def lapla_loss(pred_pos: torch.Tensor, mesh: Mesh, ltype="RMSE") -> torch.Tensor:
    # laplacian loss for output meshes 
    v2v = mesh.v2v_mat.to(pred_pos.device)
    v_dims = mesh.v_dims.reshape(-1, 1).to(pred_pos.device)
    lap_pos = torch.sparse.mm(v2v, pred_pos) / v_dims
    lap_diff = torch.sum((pred_pos - lap_pos) ** 2, dim=1)
    if ltype == "MAE":
        lap_diff = torch.sqrt(lap_diff + 1.0e-12)
        lap_loss = torch.sum(lap_diff) / len(lap_diff)
    elif ltype == "RMSE":
        lap_loss = torch.sum(lap_diff) / len(lap_diff)
        lap_loss = torch.sqrt(lap_loss + 1.0e-12)
    else:
        exit()

    return lap_loss

def fac_loss(pred_norm: Union[torch.Tensor, np.ndarray], real_norm: Union[torch.Tensor, np.ndarray], ltype="MAE") -> torch.Tensor:#l1mae
    # loss for (vertex, face) normal 
    if type(pred_norm) == np.ndarray:
        pred_norm = torch.from_numpy(pred_norm)
    if type(real_norm) == np.ndarray:
        real_norm = torch.from_numpy(real_norm).to(pred_norm.device)
    
    if ltype == "MAE":
        norm_diff = torch.sum((pred_norm - real_norm) ** 2, dim=1)
        loss = torch.sqrt(norm_diff + 1e-12)
        loss = torch.sum(loss) / len(loss)
    elif ltype == "L1":
        norm_diff = torch.sum(torch.abs(pred_norm - real_norm), dim=1)
        loss = torch.sum(norm_diff) / len(norm_diff)
    elif ltype == "RMSE":
        norm_diff = torch.sum((pred_norm - real_norm) ** 2, dim=1)
        loss = torch.sum(norm_diff) / len(norm_diff)
        loss = torch.sqrt(loss + 1e-12)
    elif ltype == "L2":
        norm_diff = torch.sum(torch.abs(pred_norm - real_norm), dim=1)
        loss = torch.sum(norm_diff ** 2) / len(norm_diff)
        loss = torch.sqrt(loss + 1e-12)
    else:
        exit()

    return loss

def squared_norm(x, dim=None, keepdim=False):
    return torch.sum(x * x, dim=dim, keepdim=keepdim)

def norm(x, eps=1.0e-12, dim=None, keepdim=False):
    return torch.sqrt(squared_norm(x, dim=dim, keepdim=keepdim) + eps)

def bnf_loss(pos: torch.Tensor, fn: torch.Tensor, mesh: Mesh, ltype="MAE", loop=5) -> torch.Tensor:
    # bilateral loss for face normal
    if type(pos) == np.ndarray:
        pos = torch.from_numpy(pos).to(fn.device)
    else:
        pos = pos.detach()
    fc = torch.sum(pos[mesh.faces], 1) / 3.0
    fa = torch.cross(pos[mesh.faces[:, 1]] - pos[mesh.faces[:, 0]], pos[mesh.faces[:, 2]] - pos[mesh.faces[:, 0]])
    fa = 0.5 * torch.sqrt(torch.sum(fa**2, axis=1) + 1.0e-12)
    
    f2f = torch.from_numpy(mesh.f2f).long().to(fn.device)
    no_neig = 1.0 * (f2f != -1)
    
    neig_fc = fc[f2f]
    neig_fa = fa[f2f] * no_neig
    fc0_tile = fc.reshape(-1, 1, 3)
    fc_dist = squared_norm(neig_fc - fc0_tile, dim=2)
    sigma_c = torch.sum(torch.sqrt(fc_dist + 1.0e-12)) / (fc_dist.shape[0] * fc_dist.shape[1])

    new_fn = fn
    for i in range(loop):
        neig_fn = new_fn[f2f]
        fn0_tile = new_fn.reshape(-1, 1, 3)
        fn_dist = squared_norm(neig_fn - fn0_tile, dim=2)
        sigma_s = 0.3
        wc = torch.exp(-1.0 * fc_dist / (2 * (sigma_c ** 2)))
        ws = torch.exp(-1.0 * fn_dist / (2 * (sigma_s ** 2)))
        
        W = torch.stack([wc*ws*neig_fa, wc*ws*neig_fa, wc*ws*neig_fa], dim=2)

        new_fn = torch.sum(W * neig_fn, dim=1)
        new_fn = new_fn / (norm(new_fn, dim=1, keepdim=True) + 1.0e-12)

    if ltype == "L1":
        bnf_diff = torch.sum((new_fn - fn) ** 2, dim=1)
        bnf_diff = torch.sqrt(bnf_diff + 1.0e-12)
        loss = torch.sum(bnf_diff) / len(bnf_diff)
    elif ltype == "MAE":
        bnf_diff = torch.sum(torch.abs(new_fn - fn), dim=1)
        loss = torch.sum(bnf_diff) / len(bnf_diff)
    elif ltype == "RMSE":
        bnf_diff = torch.sum((new_fn - fn) ** 2, dim=1)
        loss = torch.sum(bnf_diff) / len(bnf_diff)
        loss = torch.sqrt(loss + 1.0e-12)
    elif ltype == "L2":
        bnf_diff = torch.sum(torch.abs(new_fn - fn), dim=1)
        loss = torch.sum(bnf_diff ** 2) / len(bnf_diff)
        loss = torch.sqrt(loss ** 2 + 1.0e-12)
    else:
        exit()
    
    return loss, new_fn

def ver_fac_loss(pos: Union[torch.Tensor, np.ndarray], norm: Union[torch.Tensor, np.ndarray], mesh: Mesh, ltype="MAE") -> torch.Tensor:
    # loss between vertex position and face normal
    if type(pos) == np.ndarray:
        pos = torch.from_numpy(pos)
    if type(norm) == np.ndarray:
        norm = torch.from_numpy(norm).to(pos.device)
    #fc: centroid
    fc = torch.sum(pos[mesh.faces], 1) / 3.0
    #pc: position differences between each vertex in a face and the face centroid
    pc = pos[mesh.faces] - fc.reshape(-1, 1, 3)
    dot_f2v = torch.abs(torch.sum(pc * norm.reshape(-1, 1, 3), dim=2))
    mat_vals = dot_f2v.reshape(-1)

    if ltype == "MAE":
        loss = torch.sum(mat_vals) / len(mesh.vs)
    elif ltype == "RMSE":
        loss = torch.sum(mat_vals ** 2) / len(mat_vals)
        loss = torch.sqrt(loss + 1.0e-6)
    else:
        exit()

    return loss


def aae(norm1: Union[np.ndarray, torch.Tensor], norm2: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    # average angular error for (face, vertex) normals 
    if type(norm1) == torch.Tensor:
        norm1 = norm1.to("cpu").detach().numpy().copy()
    if type(norm2) == torch.Tensor:
        norm2 = norm2.to("cpu").detach().numpy().copy()

    inner = np.sum(norm1 * norm2, 1)
    sad = np.rad2deg(np.arccos(np.clip(inner, -1.0, 1.0)))
    aae = np.sum(sad) / len(sad)

    return aae

