# Airway mesh smoothing 
**This repository contains codes for the mesh (random shapes of meshes airway meshes) smoothing using graph-convolutional neural networks.**

## Files
The repository contains the following files:
- "train_optuna.py": This code trains two graph neural networks on the input mesh to filter vertex positions and facet normals. The hyper-parameters (iterations, optimizer, learning rate, type of loss functions,...) are optimized using the Optuna library.
- "trainAirway.py": This code is trained for airway meshes without ground truth. It loads a random model (unsmoothed and ground truth mesh - .obj file). The outputs are quantitatively evaluated by the average angular error (AAE) of face normals against the ground truth shapes.  
 - The "codes" folder contains files related to data generator, losses and networks.
 + "compute_mesh.py": constructing and manipulating meshes, computing geometric properties (edges, vertex-to-edge associations, and edge-to-face associations) and building various matrices for mesh (computes the vertex normals, calculates the normal vector for each face, calculates the center point of each face; constructs the vertex-to-face sparse matrix and the face-to-face (1-ring) matrix and computes the vertex-to-vertex adjacency matrix,...)
 + "dataset_gen.py": generates a dataset for mesh-related information
 + "losses.py": loss functions
 + "networks.py": GCN models

## Usage
* **Training:** To train GNN model, run the following command:
    - > python train_optuna.py 
    - > python trainAirway.py "<path_to_save_data>" 
* **Evaluation:** The trained file will output the smoothed meshes and the average angular errors. Airway meshes are evaluated based on diameters.


## Dependencies
The code requires the following Python libraries:

* 'pandas'
* 'numpy'
* 'matplotlib'
* 'scikit-learn'
* 'torch'
* 'torch-scatter'
* 'torch-sparse'
* 'torch-cluster'
* 'torch-geometric'
* 'pymeshlab'
* 'tqdm'


## License
This code is released under the MLFM Lab.
Some codes in this repository are modified from Dual deep mesh prior.
Paper: "Graph-Convolutional Neural Network-based Surface Mesh Smoothing for Human Airways".



