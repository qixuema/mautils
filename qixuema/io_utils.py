import numpy as np
import yaml

def read_vertices_obj(filename):
    vertices = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if parts[0] == 'v':  # Vertex description
                # Convert x, y, z to float and append to the vertices list
                vertices.append([float(coord) for coord in parts[1:4]])
    return np.array(vertices)

def read_obj_file(file_path):
    vertices = []
    faces = []
    lines = []
    vertice_normals = []

    with open(file_path, 'r') as file:
        for line in file:
            components = line.strip().split()

            if not components:
                continue

            if components[0] == 'v':
                # Parse vertex coordinates
                vertex = tuple(map(float, components[1:4]))
                vertices.append(vertex)
            elif components[0] == 'f':
                # Parse face indices (assuming triangular faces)
                face = tuple(map(int, components[1:4]))
                # face = [list(map(int, p.split('/'))) for p in components[1:]]
                faces.append(face)
                
            elif components[0] == 'l':
                # Parse face indices (assuming triangular faces)
                line = tuple(map(int, components[1:3]))
                lines.append(line)
            
            elif components[0] == 'vn':
                vertice_normal = tuple(map(float, components[1:4]))
                vertice_normals.append(vertice_normal)

    return {
        'vertices': np.array(vertices), 
        'faces': np.array(faces) - 1, 
        'lines': np.array(lines) - 1,
        'v_normals': np.array(vertice_normals)}

def write_obj_file(file_path, vertices, faces=None, vtx_colors=None, is_line=False, is_point=False):
    """
    Save a simple OBJ file with vertices and faces.

    Parameters:
    file_path (str): The path to save the OBJ file.
    vertices (list of tuples): A list of vertices, each vertex is a tuple (x, y, z).
    faces (list of tuples): A list of faces, each face is a tuple of vertex indices (1-based).
    """
    
    # 先转换为numpy数组
    vertices = np.array(vertices) if isinstance(vertices, list) else vertices
    faces = np.array(faces) if isinstance(faces, list) else faces
    
    if faces is not None:
        if faces.shape[1] == 2:
            is_line = True

    with open(file_path, 'w') as file:
        # Write vertices
        if vtx_colors is None:
            for vertex in vertices:
                    file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        else:
            for vertex, vtx_color in zip(vertices, vtx_colors):
                file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} {vtx_color[0]} {vtx_color[1]} {vtx_color[2]}\n")

        if is_point:
            return
        
        # Write faces
        if faces is None:
            return
        
        for face in faces:
            face_str = ' '.join([str(index + 1) for index in face])
            if is_line:
                file.write(f"l {face_str}\n")
            else:
                file.write(f"f {face_str}\n")

# Load configuration from YAML
def load_yaml(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
            

if __name__=="__main__":
    # Example usage
    vertices = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    faces = [(1, 2, 3)]
    write_obj_file('example.obj', vertices, faces)


