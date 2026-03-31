import meshio
import numpy as np

# ---- Load the mesh ----
filename = "plate_with_hole.xdmf"  # change to your file
mesh = meshio.read(filename)

# ---- Extract node positions ----
points = mesh.points  # shape (N, 3)
num_nodes = points.shape[0]

print(f"Number of nodes: {num_nodes}")
print("First 5 node positions:")
print(points[:5])

# ---- Extract connectivity ----
# mesh.cells is a list of (cell_type, array)
# For edges, look for "line"

edges = None

for cell_block in mesh.cells:
    if cell_block.type == "line":
        edges = cell_block.data
        break

if edges is None:
    raise ValueError("No edge (line) elements found in the mesh.")

num_edges = edges.shape[0]

print(f"\nNumber of edges: {num_edges}")
print("First 5 edges (node indices):")
print(edges[:5])

# ---- Optional: build explicit edge list ----
edge_list = [(int(i), int(j)) for i, j in edges]

# ---- Optional: node IDs ----
node_ids = np.arange(num_nodes)

# ---- Example: access a specific edge ----
edge_id = 0
n1, n2 = edge_list[edge_id]
print(f"\nEdge {edge_id} connects node {n1} to node {n2}")
print(f"Coordinates:")
print(points[n1], points[n2])
