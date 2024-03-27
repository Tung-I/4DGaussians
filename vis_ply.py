import open3d as o3d  

def main():
    fname = "/home/ubuntu/datasets/dynerf/cook_spinach/colmap/dense/workspace/fused.ply"
    cloud = o3d.io.read_point_cloud(fname) # Read point cloud
    o3d.visualization.draw_geometries([cloud])    # Visualize point cloud      

if __name__ == "__main__":
    main()