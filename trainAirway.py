import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
import torch.nn as nn
import os, sys
from tqdm import tqdm
import codes.losses as Losses
import codes.dataset_gen as DataGen
from codes.compute_mesh import Mesh
from codes.networks import GNNVer, GNNFac

# airway mesh without GT
def trainairway(input_mesh):
    ver_lr = 0.01
    fac_lr = 0.01
    itera = 250
    k1 = 3
    k2 = 0
    k3 = 3
    k4 = 4
    k5 = 2
    grad_crip = 0.8
    bfnloop = 5
    gpu = 0

    #dataset
    mesh_dic, dataset = DataGen.create_dataset(input_mesh)
    mesh_name = mesh_dic["mesh_name"]
    n_mesh, o1_mesh = mesh_dic["n_mesh"], mesh_dic["o1_mesh"]

    #network
    device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
    vernet = GNNVer(device).to(device)
    facnet = GNNFac(device).to(device)
    optimizer_ver = torch.optim.Adam(vernet.parameters(), lr=ver_lr)
    optimizer_fac = torch.optim.Adam(facnet.parameters(), lr=fac_lr)

    os.makedirs("input_data/" + mesh_name + "/outputmesh", exist_ok=True)

    #training
    with tqdm(total=itera) as pbar:
        for epoch in range(1, itera+1):
            vernet.train()
            facnet.train()
            optimizer_ver.zero_grad()
            optimizer_fac.zero_grad()

            ver = vernet(dataset)
            loss_ver1 = Losses.ver_loss(ver, n_mesh.vs)
            loss_ver2 = Losses.lapla_loss(ver, n_mesh)

            fac = facnet(dataset)
            loss_fac1 = Losses.fac_loss(fac, n_mesh.fn)
            loss_fac2, new_fn = Losses.bnf_loss(ver, fac, n_mesh, loop=bfnloop)

            if epoch <= 50:
                loss_fac2 = loss_fac2 * 0.0

            loss_ver3 = Losses.ver_fac_loss(ver, fac, n_mesh)

            loss = k1 * loss_ver1 + k2 * loss_ver2 + k3 * loss_fac1 + k4 * loss_fac2 + k5 * loss_ver3
            loss.backward()
            nn.utils.clip_grad_norm_(facnet.parameters(), grad_crip)
            optimizer_ver.step()
            optimizer_fac.step()

            pbar.set_description("Epoch {}".format(epoch))
            pbar.set_postfix({"loss": loss.item()})

            if epoch % 10 == 0:
                o1_mesh.vs = ver.to('cpu').detach().numpy().copy()
                o_path = "input_data/" + mesh_name + "/outputmesh/" + str(epoch) + "_amsl.obj"
                Mesh.save(o1_mesh, o_path)

            pbar.update(1)

if __name__ == "__main__":
    input_mesh = sys.argv[1]
    trainairway(input_mesh)
    # input_mesh  = "input_data/10147"