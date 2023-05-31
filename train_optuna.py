import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv

import codes.losses as Losses
import codes.dataset_gen as DataGen
from codes.compute_mesh import Mesh
from tqdm import tqdm

#for visualization
import plotly
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


#GNN models
class GNNVer(nn.Module):
    def __init__(self, device, trial: optuna.Trial):
        super(GNNVer, self).__init__()
        self.device = device
        
        feu = [16, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 16, 3]
        neuron1 = trial.suggest_categorical("neuron1", [8, 16, 28, 32])

        self.conv1  = GCNConv(feu[0], feu[1]) 
        self.conv2  = GCNConv(feu[1], feu[2])
        self.conv3  = GCNConv(feu[2], feu[3])
        self.conv4  = GCNConv(feu[3], feu[4])
        self.conv5  = GCNConv(feu[4], feu[5])
        self.conv6  = GCNConv(feu[5], feu[6])
        self.conv7  = GCNConv(feu[6], feu[7])
        self.conv8  = GCNConv(feu[7], feu[8])
        self.conv9  = GCNConv(feu[8], feu[9])
        self.conv10 = GCNConv(feu[9], feu[10])
        self.conv11 = GCNConv(feu[10], feu[11])
        self.conv12 = GCNConv(feu[11], feu[12])
        
        self.linear1 = nn.Linear(feu[12], neuron1) 
        self.linear2 = nn.Linear(neuron1, feu[14])
    
        self.bn1 = nn.BatchNorm1d(feu[1])
        self.bn2 = nn.BatchNorm1d(feu[2])
        self.bn3 = nn.BatchNorm1d(feu[3])
        self.bn4 = nn.BatchNorm1d(feu[4])
        self.bn5 = nn.BatchNorm1d(feu[5])
        self.bn6 = nn.BatchNorm1d(feu[6])
        self.bn7 = nn.BatchNorm1d(feu[7])
        self.bn8 = nn.BatchNorm1d(feu[8])
        self.bn9 = nn.BatchNorm1d(feu[9])
        self.bn10 = nn.BatchNorm1d(feu[10])
        self.bn11 = nn.BatchNorm1d(feu[11])
        self.bn12 = nn.BatchNorm1d(feu[12])
        self.l_relu = nn.LeakyReLU()
            
    def forward(self, data):
        z1, x_pos, edge_index = data.z1.to(self.device), data.x_pos.to(self.device), data.edge_index.to(self.device)
        dx = self.l_relu(self.bn1(self.conv1(z1, edge_index)))
        dx = self.l_relu(self.bn2(self.conv2(dx, edge_index)))
        dx = self.l_relu(self.bn3(self.conv3(dx, edge_index)))
        dx = self.l_relu(self.bn4(self.conv4(dx, edge_index)))
        dx = self.l_relu(self.bn5(self.conv5(dx, edge_index)))
        dx = self.l_relu(self.bn6(self.conv6(dx, edge_index)))
        dx = self.l_relu(self.bn7(self.conv7(dx, edge_index)))
        dx = self.l_relu(self.bn8(self.conv8(dx, edge_index)))
        dx = self.l_relu(self.bn9(self.conv9(dx, edge_index)))
        dx = self.l_relu(self.bn10(self.conv10(dx, edge_index)))
        dx = self.l_relu(self.bn11(self.conv11(dx, edge_index)))
        dx = self.l_relu(self.bn12(self.conv12(dx, edge_index)))
        
        dx = self.l_relu(self.linear1(dx))
        dx = self.linear2(dx)
        
        return x_pos + dx

class GNNFac(nn.Module):
    def __init__(self, device, trial: optuna.Trial):
        super(GNNFac, self).__init__()
        self.device = device
        
        feu = [7, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 16, 3]
        neuron1 = trial.suggest_categorical("neuron2", [8, 16, 28, 32])

        self.conv1  = GCNConv(feu[0], feu[1])
        self.conv2  = GCNConv(feu[1], feu[2])
        self.conv3  = GCNConv(feu[2], feu[3])
        self.conv4  = GCNConv(feu[3], feu[4])
        self.conv5  = GCNConv(feu[4], feu[5])
        self.conv6  = GCNConv(feu[5], feu[6])
        self.conv7  = GCNConv(feu[6], feu[7])
        self.conv8  = GCNConv(feu[7], feu[8])
        self.conv9  = GCNConv(feu[8], feu[9])
        self.conv10 = GCNConv(feu[9], feu[10])
        self.conv11 = GCNConv(feu[10], feu[11])
        self.conv12 = GCNConv(feu[11], feu[12])

        self.linear1 = nn.Linear(feu[12], neuron1)
        self.linear2 = nn.Linear(neuron1, feu[14])
    
        self.bn1 = nn.BatchNorm1d(feu[1])
        self.bn2 = nn.BatchNorm1d(feu[2])
        self.bn3 = nn.BatchNorm1d(feu[3])
        self.bn4 = nn.BatchNorm1d(feu[4])
        self.bn5 = nn.BatchNorm1d(feu[5])
        self.bn6 = nn.BatchNorm1d(feu[6])
        self.bn7 = nn.BatchNorm1d(feu[7])
        self.bn8 = nn.BatchNorm1d(feu[8])
        self.bn9 = nn.BatchNorm1d(feu[9])
        self.bn10 = nn.BatchNorm1d(feu[10])
        self.bn11 = nn.BatchNorm1d(feu[11])
        self.bn12 = nn.BatchNorm1d(feu[12])

        self.l_relu = nn.LeakyReLU()            

    def forward(self, data):

        z2, x_pos, face_index = data.z2.to(self.device), data.x_pos.to(self.device), data.face_index.to(self.device)
        dx = self.l_relu(self.bn1(self.conv1(z2, face_index)))
        dx = self.l_relu(self.bn2(self.conv2(dx, face_index)))
        dx = self.l_relu(self.bn3(self.conv3(dx, face_index)))
        dx = self.l_relu(self.bn4(self.conv4(dx, face_index)))
        dx = self.l_relu(self.bn5(self.conv5(dx, face_index)))
        dx = self.l_relu(self.bn6(self.conv6(dx, face_index)))
        dx = self.l_relu(self.bn7(self.conv7(dx, face_index)))
        dx = self.l_relu(self.bn8(self.conv8(dx, face_index)))
        dx = self.l_relu(self.bn9(self.conv9(dx, face_index)))
        dx = self.l_relu(self.bn10(self.conv10(dx, face_index)))
        dx = self.l_relu(self.bn11(self.conv11(dx, face_index)))
        dx = self.l_relu(self.bn12(self.conv12(dx, face_index)))
        
        dx = self.l_relu(self.linear1(dx))
        dx = torch.tanh(self.linear2(dx))

        dx_norm = torch.reciprocal(torch.norm(dx, dim=1, keepdim=True).expand(-1, 3) + 1.0e-12)
        x = torch.mul(dx, dx_norm)
        return x

#hyper-parameter tunning using optuna
def optuna_model(trial):

    gpu = 0
    #networks
    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    gnnver = GNNVer(device, trial).to(device)
    gnnfac = GNNFac(device, trial).to(device)

    # Set paramaters
    lamda_1 = trial.suggest_int("lamda_1", 1, 5) 
    lamda_2 = trial.suggest_int("lamda_2", 0, 5)
    lamda_3 = trial.suggest_int("lamda_3", 1, 5)
    lamda_4 = trial.suggest_int("lamda_4", 1, 5)
    lamda_5 = trial.suggest_int("lamda_5", 1, 5)

    iteration = trial.suggest_categorical("iteration", [1300, 1310, 1320, 1330, 1340, 1350, 1360, 1370, 1380, 1390,
                                                        1400, 1410, 1420, 1430, 1440, 1450, 1460, 1470, 1480, 1490, 1500])
                                                        
    loss_ver = trial.suggest_categorical("loss_ver", ["RMSE","MAE"])
    loss_lap = trial.suggest_categorical("loss_lap", ["RMSE","MAE"])
    loss_fac = trial.suggest_categorical("loss_fac", ["RMSE","MAE", "L1", "L2"])
    loss_bnf = trial.suggest_categorical("loss_bnf", ["RMSE","MAE", "L1", "L2"])
    loss_con = trial.suggest_categorical("loss_con", ["RMSE","MAE"])

    lr_ver = trial.suggest_categorical("lr_ver", [0.001, 0.002, 0.005, 0.01, 0.02])
    lr_fac = trial.suggest_categorical("lr_fac", [0.001, 0.002, 0.005, 0.01, 0.02])

    optimizer_name = trial.suggest_categorical("optimizer_name", ["SGD", "Adam"])
    optimizer_ver = getattr(optim, optimizer_name)(gnnver.parameters(), lr=lr_ver)
    optimizer_fac = getattr(optim, optimizer_name)(gnnfac.parameters(), lr=lr_fac)

    #input data
    input_model  = 'input_data/block/'
    grad_crip = 0.8
    bnfloop = 5
    
    #dataset
    mesh_dic, dataset = DataGen.create_dataset(input_model)
    mesh_name = mesh_dic["mesh_name"]
    gt_mesh, n_mesh, o1_mesh = mesh_dic["gt_mesh"], mesh_dic["n_mesh"], mesh_dic["o1_mesh"]

    optimizer_ver = optimizer_ver
    optimizer_fac = optimizer_fac
    
    os.makedirs(input_model + "/output_tuning_optuna", exist_ok=True)

    #initial aae
    init_aae = aae_value = Losses.aae(n_mesh.fn, gt_mesh.fn)
#     print("initial_aae: {:.3f}".format(init_aae))

    #training
    with tqdm(total=iteration) as pbar:
        aae_value_min = []
        for epoch in range(1, iteration+1):
            gnnver.train()
            gnnfac.train()
            optimizer_ver.zero_grad()
            optimizer_fac.zero_grad()

            ver = gnnver(dataset)
            loss_ver1 = Losses.ver_loss(ver, n_mesh.vs, ltype = loss_ver)
            loss_ver2 = Losses.lapla_loss(ver, n_mesh, ltype = loss_lap)

            fac = gnnfac(dataset)
            loss_fac1 = Losses.fac_loss(fac, n_mesh.fn, ltype = loss_fac)
            loss_fac2, _ = Losses.bnf_loss(ver, fac, n_mesh, loop=bnfloop, ltype = loss_bnf)
            if epoch <= 100:
                loss_fac2 = loss_fac2 * 0.0

            loss_ver3 = Losses.ver_fac_loss(ver, fac, n_mesh, ltype = loss_con)

            loss = lamda_1*loss_ver1 + lamda_2*loss_ver2 + lamda_3*loss_fac1 + lamda_4*loss_fac2 + lamda_5*loss_ver3
            loss.backward()
            nn.utils.clip_grad_norm_(gnnfac.parameters(), grad_crip)
            optimizer_ver.step()
            optimizer_fac.step()

            pbar.set_description("Epoch {}".format(epoch))
            pbar.set_postfix({"loss": loss.item()})

            vs_update = False

            if epoch % 10 == 0:
                new_ver = ver.to("cpu").detach().numpy().copy()
                o1_mesh.vs = new_ver
                Mesh.compute_face_normals(o1_mesh)
                Mesh.compute_vert_normals(o1_mesh)

                aae_value = Losses.aae(o1_mesh.fn, gt_mesh.fn)
                # o_path = input_model + "/output_tuning_optuna/" + str(epoch) + "_amsl={:.3f}.obj".format(aae_value)
                aae_value_min.append(aae_value)
                aae_value_min1 = np.min(aae_value_min)
                # Mesh.save(o1_mesh, o_path)                

                if vs_update:
                    updated_pos = Losses.vertex_updating(ver, fac, o1_mesh, loop=15)
                    o1_mesh.vs = updated_pos.to("cpu").detach().numpy().copy()
                    Mesh.compute_face_normals(o1_mesh)
                    updated_aae = Losses.aae(o1_mesh.fn, gt_mesh.fn)
                    u_path = input_model + "/output_tuning_optuna/" + str(epoch) + "_amsl_updated={:.3f}.obj".format(updated_aae)
                    Mesh.save(o1_mesh, u_path)

            pbar.update(1)
    # print("final_aae: {:.3f}".format(aae_value))
    # print("final_aae_min: {:.3f}".format(aae_value_min1))
    
    return aae_value_min1

def main():
    # Run Optuna
    start=time.time()
    study = optuna.create_study(study_name="Hyperparameter_Optimization_block", direction='minimize')
    study.optimize(optuna_model, n_trials=2)
    print('It takes %s minutes' % ((time.time() - start)/60))

    study.best_params

    df = study.trials_dataframe()

    #visualization of accuracy vs #trials 
    import matplotlib as plt
    ax = df['value'].plot()
    ax.set_xlabel('Number of trials')
    ax.set_ylabel('Accuracy by AAE')

    #optuna visualization results
    # https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html

    plot_optimization_history(study)
    plot_parallel_coordinate(study)
    plot_parallel_coordinate(study, params=["neuron1", "neuron2","iteration","lamda_1", "lamda_2","lamda_3","lamda_4","lamda_5","loss_ver","loss_lap","loss_fac",
                                        "loss_bnf","loss_con","lr_ver","lr_fac","optimizer_name"])
    plot_slice(study)
    plot_slice(study, params=["lamda_1", "lamda_2","lamda_3","lamda_4","lamda_5"])
    plot_param_importances(study)
    plot_edf(study)

if __name__ == "__main__":
    main()
