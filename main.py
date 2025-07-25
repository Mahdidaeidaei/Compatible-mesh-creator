import trimesh
import numpy as np
import os
import multiprocessing as mp
from scipy.spatial.distance import cdist
import argparse

from visualizer import load_and_plot_npy_files

# Function to subdivide triangles
def subdivide_triangle(v1_idx, v2_idx, v3_idx, vertices):
    """ Subdivides a triangle into 4 smaller triangles. """
    v1, v2, v3 = np.array(vertices[v1_idx]), np.array(vertices[v2_idx]), np.array(vertices[v3_idx])
    mid1, mid2, mid3 = (v1 + v2) / 2, (v2 + v3) / 2, (v3 + v1) / 2

    i1, i2, i3 = len(vertices), len(vertices) + 1, len(vertices) + 2
    vertices.extend([mid1.tolist(), mid2.tolist(), mid3.tolist()])

    return [[v1_idx, i1, i3], [i1, v2_idx, i2], [i3, i2, v3_idx], [i1, i2, i3]]

# Function to refine mesh
def refine_mesh(mesh, max_edge_length):
    """Iteratively refine the mesh until no element has an edge longer than max_edge_length."""
    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()

    def longest_edge(face):
        """Compute the longest edge in a given triangular face."""
        v1, v2, v3 = np.array(vertices[face[0]]), np.array(vertices[face[1]]), np.array(vertices[face[2]])
        return max(np.linalg.norm(v1 - v2), np.linalg.norm(v2 - v3), np.linalg.norm(v3 - v1))
    
    subdivided = True
    while subdivided:  # Continue refining until all edges are within limit
        new_faces = []
        subdivided = False
        
        for face in faces:
            if longest_edge(face) > max_edge_length:
                new_faces.extend(subdivide_triangle(*face, vertices))  # Subdivide if needed
                subdivided = True  # Mark that at least one face was subdivided
            else:
                new_faces.append(face)  # Keep small elements unchanged
        
        faces = new_faces  # Update faces with refined ones

    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))


# Function to compute signed distance from a point to a plane
def signed_distance(point, plane_normal, plane_point):
    return np.dot(plane_normal, point - plane_point)

# Function to filter mesh faces where all vertices have y ≥ 0
def filter_mesh_faces(mesh):
    filtered_faces = [face for face in mesh.faces if np.all(np.logical_and(mesh.vertices[face][:, 1] >= -10, mesh.vertices[face][:, 0] <= 10))]
    return trimesh.Trimesh(vertices=mesh.vertices, faces=filtered_faces)

# Nearest-neighbor sorting
def nearest_neighbor_sort(points):
    start_idx = np.argmin(points[:, 1])
    sorted_points, remaining_points = [points[start_idx]], np.delete(points, start_idx, axis=0).tolist()
    
    while remaining_points:
        last_point = sorted_points[-1]
        distances = cdist([last_point], remaining_points)[0]
        sorted_points.append(remaining_points.pop(np.argmin(distances)))
    
    return np.array(sorted_points)

# Compute curve length
def calculate_curve_length(points):
    return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

# Function to interpolate points along the curve
def interpolate_points_along_curve(points, pace):
    new_points, remaining_dist = [points[0]], pace

    for i in range(1, len(points)):
        p1, p2 = points[i - 1], points[i]
        segment_length = np.linalg.norm(p2 - p1)

        while remaining_dist < segment_length:
            new_point = p1 + (remaining_dist / segment_length) * (p2 - p1)
            new_points.append(new_point)
            remaining_dist += pace

        remaining_dist -= segment_length

    return np.array(new_points)

# Function to process a single plane slicing
def process_plane(angle, filtered_mesh):
    plane_normal = np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0])
    plane_point, intersection_points = np.array([0, 0, 0]), []

    for face in filtered_mesh.faces:
        vertices = filtered_mesh.vertices[face]
        distances = [signed_distance(v, plane_normal, plane_point) for v in vertices]

        for i in range(3):  # Check each edge of the triangle
            v1, v2, d1, d2 = vertices[i], vertices[(i + 1) % 3], distances[i], distances[(i + 1) % 3]
            if np.sign(d1) != np.sign(d2):  # Edge crosses the plane
                t = d1 / (d1 - d2)
                intersection_point = v1 + t * (v2 - v1)
                
                if intersection_point[1] > 0:  # Only append if y > 0
                    intersection_points.append(intersection_point)

    if not intersection_points:
        return angle, None

    intersection_points = np.array(intersection_points)
    sorted_points = nearest_neighbor_sort(intersection_points)

    sorted_points = np.vstack([[0, 0, sorted_points[0][2]], sorted_points, [0, 0, sorted_points[-1][2]]])
    curve_length, pace = calculate_curve_length(sorted_points), calculate_curve_length(sorted_points) / 200
    interpolated_points = np.round(interpolate_points_along_curve(sorted_points, pace), 3)

    expected_last_point = np.array([0, 0, interpolated_points[-1][2]])
    if not np.any(np.all(np.isclose(interpolated_points, expected_last_point, atol=1e-3), axis=1)):
        interpolated_points = np.vstack([interpolated_points, expected_last_point])

    return angle, interpolated_points

# Function to process a single STL file
def process_stl(file_path, output_folder, max_edge_length):
    mesh = trimesh.load_mesh(file_path)
    
    refined_mesh = refine_mesh(mesh, max_edge_length)
    filtered_mesh = filter_mesh_faces(refined_mesh)

    num_workers = mp.cpu_count()
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(process_plane, [(angle, filtered_mesh) for angle in np.arange(1.5,90,3)])

    results.sort(key=lambda x: x[0])
    all_interpolated_points = np.array([r[1] for r in results], dtype=object)

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    np.save(os.path.join(output_folder, f"{file_name}_interpolated_curves.npy"), all_interpolated_points)
    print(f"Saved: {file_name}_interpolated_curves.npy")

# Function to process all STL files in a folder
def process_all_stl_files(input_folder, output_folder, max_edge_length):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    stl_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".stl")]
    for stl_file in stl_files:
        process_stl(stl_file, output_folder, max_edge_length)

# Example usage
if __name__ == "__main__":
    input_folder = "database"  # Change to your STL folder
    output_folder = "npyfiles"   # Change to desired output location

    parser = argparse.ArgumentParser(description="creates compatible mesh")
    parser.add_argument("--e", type=float, default=1, help="edge length treshold for mesh refinement")

    args = parser.parse_args()

    process_all_stl_files(input_folder, output_folder, args.e)
    load_and_plot_npy_files()  # Call the function to load and plot .npy files
