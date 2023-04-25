import matplotlib.pyplot as plt
from rdkit import Chem
from scipy.sparse import coo_matrix
import torch
import matplotlib as mpl
from torch_geometric.data import Data
from IPython.display import SVG, display
from rdkit.Chem.Draw import rdMolDraw2D
from FFiNet.data_pipeline.data_generating import subgraph_index


def molecule_attn_visulize(smiles, target_index=None, edge_attn=None, angle_attn=None, dihedral_attn=None, 
                        axial_attn=None):
        
        target_index= torch.tensor(target_index, dtype=torch.long)       
        mol = Chem.MolFromSmiles(smiles)

        norm = mpl.colors.Normalize(vmin=0,vmax=1)
        cmap = mpl.cm.get_cmap('autumn')
        plt_colors = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        atom_index = []
        atom_color = {}
        atom_size = {}
        
        # axial attention
        axial_attn_target = axial_attn[:, target_index.item(), :]

        # generate edge index
        adj = Chem.GetAdjacencyMatrix(mol)
        coo_adj = coo_matrix(adj)
        edge_index = torch.as_tensor([coo_adj.row, coo_adj.col], dtype=torch.long)
        edge_src_index, edge_dst_index = edge_index
        edge_plot_index = torch.nonzero(edge_dst_index == target_index).squeeze(-1)
        for i in edge_plot_index:
            atom_index.append(edge_src_index[i].item())
            atom_color[edge_src_index[i].item()] = plt_colors.to_rgba(float(edge_attn[i].item()))
            atom_size[edge_src_index[i].item()] = 0.5 * axial_attn_target[0].item()
    
        # generate angle index
        data = Data(edge_index=edge_index)
        angle_index = subgraph_index(data)
        angle_src_index, _, angle_dst_index = angle_index
        angle_plot_index = torch.nonzero(angle_dst_index == target_index).squeeze(-1)
        for i in angle_plot_index:
            atom_index.append(angle_src_index[i].item())
            atom_color[angle_src_index[i].item()] = plt_colors.to_rgba(float(angle_attn[i].item()))
            atom_size[angle_src_index[i].item()] = 0.5 * axial_attn_target[1].item()
    
        # generate angle index
        dihedral_index = subgraph_index(data, num_nodes=4, target_edges=[(0, 1), (1, 2), (2, 3)])
        dihedral_src_index, _, _, dihedral_dst_index = dihedral_index
        dihedral_plot_index = torch.nonzero(dihedral_dst_index == target_index).squeeze(-1)
        for i in  dihedral_plot_index:
            atom_index.append(dihedral_src_index[i].item())
            atom_color[dihedral_src_index[i].item()] = plt_colors.to_rgba(float(dihedral_attn[i].item()))
            atom_size[dihedral_src_index[i].item()] = 0.5 * axial_attn_target[2].item()

        # target index
        atom_index.append(target_index.item())
        atom_color[target_index.item()] = (0.82, 0, 1, 0.57)
        atom_size[target_index.item()] = 0.6
        
        drawer = rdMolDraw2D.MolDraw2DSVG(300,300)
        drawer.SetFontSize(0.8)
        mol = rdMolDraw2D.PrepareMolForDrawing(mol)
        drawer.DrawMolecule(mol,highlightAtoms=atom_index,highlightBonds=[],
                        highlightAtomColors=atom_color, highlightAtomRadii=atom_size, legend=smiles)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg2 = svg.replace('svg:','')
        svg3 = SVG(svg2)
        display(svg3)