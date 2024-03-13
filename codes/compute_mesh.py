import numpy as np
import torch
import scipy as sp
from collections import Counter
from sklearn.preprocessing import normalize

class Mesh:
    def __init__(self, path):
        self.path = path
        self.vs, self.faces = self.read_file(path)
        self.compute_face_normals()
        self.compute_face_center()
        self.device = 'cpu'
        self.build_geo() 
        self.compute_ver_normals()
        self.build_v2v()
        self.build_vf()
        
    
    #Read the mesh data from a file
    def read_file(self, path):
        vs, faces = [], [] #extract vertex coordinates and face indices
        f = open(path)
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:4]])
            elif splitted_line[0] == 'f':
                face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
                assert len(face_vertex_ids) == 3
                face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind) for ind in face_vertex_ids]
                faces.append(face_vertex_ids)
        f.close()
        vs = np.asarray(vs)
        faces = np.asarray(faces, dtype=int)

        assert np.logical_and(faces >= 0, faces < len(vs)).all()
        return vs, faces

    #Building geometric information about the mesh: graph connectivity (edges), 
    # vertex-to-vertex associations, vertex-to-edge associations, and edge-to-face associations
    def build_geo(self):
        self.ve = [[] for _ in self.vs]
        self.vei = [[] for _ in self.vs]
        edge_nb = []
        sides = []
        edge2key = dict()
        edges = []
        edges_count = 0
        nb_count = []
        #iterate over each face of the mesh
        for face_id, face in enumerate(self.faces):
            faces_edges = []
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                faces_edges.append(cur_edge)
            for idx, edge in enumerate(faces_edges):
                #The vertices of each edge are sorted and stored as a tuple to 
                #ensure consistent representation regardless of the order of the vertices.
                edge = tuple(sorted(list(edge))) 
                faces_edges[idx] = edge
                if edge not in edge2key:
                    edge2key[edge] = edges_count
                    edges.append(list(edge))
                    edge_nb.append([-1, -1, -1, -1])
                    sides.append([-1, -1, -1, -1])
                    self.ve[edge[0]].append(edges_count)
                    self.ve[edge[1]].append(edges_count)
                    self.vei[edge[0]].append(0)
                    self.vei[edge[1]].append(1)
                    nb_count.append(0)
                    edges_count += 1
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
                edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
                nb_count[edge_key] += 2
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
                sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
        self.edges = np.array(edges, dtype=np.int32)#the connectivity information between vertices
        self.gemm_edges = np.array(edge_nb, dtype=np.int64)
        self.sides = np.array(sides, dtype=np.int64)
        self.edges_count = edges_count

    #Calculate the normal vector for each face using the cross product of two edge vectors
    def compute_face_normals(self):
        face_normals = np.cross(self.vs[self.faces[:, 1]] - self.vs[self.faces[:, 0]], self.vs[self.faces[:, 2]] - self.vs[self.faces[:, 0]])
        norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-24
        face_areas = 0.5 * np.sqrt((face_normals**2).sum(axis=1))
        face_normals /= norm
        self.fn, self.fa = face_normals, face_areas

    #Compute the vertex normals by aggregating the face normals that are connected to each vertex
    def compute_ver_normals(self):
        ver_normals = np.zeros((3, len(self.vs)))
        face_normals = self.fn
        faces = self.faces

        nv = len(self.vs)
        nf = len(faces)
        mat_rows = faces.reshape(-1)
        mat_cols = np.array([[i] * 3 for i in range(nf)]).reshape(-1)
        mat_vals = np.ones(len(mat_rows))
        f2v_mat = sp.sparse.csr_matrix((mat_vals, (mat_rows, mat_cols)), shape=(nv, nf))
        ver_normals = sp.sparse.csr_matrix.dot(f2v_mat, face_normals)
        ver_normals = normalize(ver_normals, norm='l2', axis=1)
        self.vn = ver_normals
    
    #Calculate the center point of each face by averaging the coordinates of its vertices
    def compute_face_center(self):
        faces = self.faces
        vs = self.vs
        self.fc = np.sum(vs[faces], 1) / 3.0
    

    #Construct the vertex-to-face sparse matrix and the face-to-face (1-ring) matrix. 
    #It iterates over each vertex and face to determine the neighboring faces and their corresponding indices.
    def build_vf(self):
        vf = [set() for _ in range(len(self.vs))]
        for i, f in enumerate(self.faces):
            vf[f[0]].add(i)
            vf[f[1]].add(i)
            vf[f[2]].add(i)
        self.vf = vf

        # build vertex-to-face sparse matrix 
        v2f_inds = [[] for _ in range(2)]
        v2f_vals = []
        v2f_areas = [[] for _ in range(len(self.vs))]
        for i in range(len(vf)):
            v2f_inds[1] += list(vf[i])
            v2f_inds[0] += [i] * len(vf[i])
            v2f_vals += (self.fc[list(vf[i])] - self.vs[i].reshape(1, -1)).tolist()
            v2f_areas[i] = np.sum(self.fa[list(vf[i])])
        self.v2f_list = [v2f_inds, v2f_vals, v2f_areas]
        
        v2f_inds = torch.tensor(v2f_inds).long()
        v2f_vals = torch.ones(v2f_inds.shape[1]).float()
        self.v2f_mat = torch.sparse.FloatTensor(v2f_inds, v2f_vals, size=torch.Size([len(self.vs), len(self.faces)]))

        # build face-to-face (1ring) matrix 
        f2f = [[] for _ in range(len(self.faces))]
        self.f_edges = [[] for _ in range(2)]
        for i, f in enumerate(self.faces):
            #For each face, it collects the neighbor face indices 
            #by combining the neighbor indices of each vertex.
            #the vf list - vertex-to-face associations, 
            #is used to obtain the neighbor face indices for each vertex.
            all_neig = list(vf[f[0]]) + list(vf[f[1]]) + list(vf[f[2]])
            #The collected neighbor face indices are checked to see if they occur twice, 
            #indicating a shared edge between the current face and another face.
            one_neig = np.array(list(Counter(all_neig).values())) == 2
            f2f_i = np.array(list(Counter(all_neig).keys()))[one_neig].tolist()
            self.f_edges[0] += len(f2f_i) * [i]
            self.f_edges[1] += f2f_i
            #the resulting neighbor face indices,
            #where each element represents the 1-ring neighbor face indices for a face.
            f2f[i] = f2f_i + (3 - len(f2f_i)) * [-1]

        self.f2f = np.array(f2f)
        self.f_edges = np.array(self.f_edges)
        

    #Compute the vertex-to-vertex adjacency matrix by using the edges information (calculating the graph connectivity information)
    def build_v2v(self):
        # compute adjacent matrix 
        edges = self.edges
        v2v_inds = edges.T
        v2v_inds = torch.from_numpy(np.concatenate([v2v_inds, v2v_inds[[1, 0]]], axis=1)).long()
        v2v_vals = torch.ones(v2v_inds.shape[1]).float()
        self.v2v_mat = torch.sparse.FloatTensor(v2v_inds, v2v_vals, size=torch.Size([len(self.vs), len(self.vs)]))
        self.v_dims = torch.sum(self.v2v_mat.to_dense(), axis=1)
        
    #Save the mesh to a file in a simple text format, storing the vertex coordinates and face indices.
    def save(self, filename):
        assert len(self.vs) > 0
        vertices = np.array(self.vs, dtype=np.float32).flatten()
        indices = np.array(self.faces, dtype=np.uint32).flatten()

        with open(filename, 'w') as fp:
            # Write positions
            for i in range(0, vertices.size, 3):
                x = vertices[i + 0]
                y = vertices[i + 1]
                z = vertices[i + 2]
                fp.write('v {0:.8f} {1:.8f} {2:.8f}\n'.format(x, y, z))

            # Write indices
            for i in range(0, len(indices), 3):
                i0 = indices[i + 0] + 1
                i1 = indices[i + 1] + 1
                i2 = indices[i + 2] + 1
                fp.write('f {0} {1} {2}\n'.format(i0, i1, i2))
    

#https://github.com/astaka-pe/Dual-DMP/blob/main/util/mesh.py