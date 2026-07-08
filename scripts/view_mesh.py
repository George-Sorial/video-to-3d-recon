"""Open an Open3D window to inspect a reconstructed mesh.

    python -m scripts.view_mesh outputs/meshes/my_room.ply
"""

import sys

import open3d as o3d


def main():
    if len(sys.argv) < 2:
        print("usage: python -m scripts.view_mesh <mesh.ply>")
        sys.exit(1)

    mesh = o3d.io.read_triangle_mesh(sys.argv[1])
    mesh.compute_vertex_normals()
    print(f"vertices: {len(mesh.vertices)}  triangles: {len(mesh.triangles)}")
    o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    main()
