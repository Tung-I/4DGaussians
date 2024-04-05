from plyfile import PlyData, PlyElement
import numpy as np

def filter_and_save_ply(input_path, output_path):
    plydata = PlyData.read(input_path)

    # Extract XYZ positions
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)  # (N, 3)
    print(f'xyz.shape: {xyz.shape}')
    
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]  # (N, 1)
    print(f'opacities.shape: {opacities.shape}')

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])  # (N, 3, 1)
    print(f'features_dc.shape: {features_dc.shape}')

    max_sh_degree = 3
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))  # (N, 3, 15)
    print(f'features_extra.shape: {features_extra.shape}')

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        print(f'attr_name: {attr_name}')
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])  # (N, 3)
    print(f'scales.shape: {scales.shape}')

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        print(f'attr_name: {attr_name}')
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])  # (N, 4)
    print(f'rots.shape: {rots.shape}')

    # Prepare position data
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    positions = np.array(list(map(tuple, xyz)), dtype=dtype)
    vertex_element = PlyElement.describe(positions, 'vertex')
    PlyData([vertex_element], text=False).write(output_path)

def replace_decoded_ply(input_ply_path, output_ply_path, decoded_ply_path):
    plydata = PlyData.read(input_ply_path)
    decoded_plydata = PlyData.read(decoded_ply_path)

    # Extract XYZ positions
    decoded_xyz = np.stack((np.asarray(decoded_plydata.elements[0]["x"]),
                            np.asarray(decoded_plydata.elements[0]["y"]),
                            np.asarray(decoded_plydata.elements[0]["z"])), axis=1)  # (N, 3)

    # Replace XYZ positions
    plydata.elements[0]["x"] = decoded_xyz[:, 0]
    plydata.elements[0]["y"] = decoded_xyz[:, 1]
    plydata.elements[0]["z"] = decoded_xyz[:, 2]

    # Save the ply file
    PlyData([plydata.elements[0]], text=False).write(output_ply_path)



# Example usage:
# input_ply_path = "/home/ubuntu/pc_compress_test/dynerf/cook_spinach/point_cloud/iteration_14000/point_cloud.ply"
# output_filtered_ply_path = "/home/ubuntu/pc_compress_test/dynerf/cook_spinach/point_cloud/iteration_14000/filtered_point_cloud.ply"
# filter_and_save_ply(input_ply_path, output_filtered_ply_path)
    

input_ply_path = "/home/ubuntu/pc_compress_test/dynerf/cook_spinach/point_cloud/iteration_14000/point_cloud.ply"
output_filtered_ply_path = "/home/ubuntu/pc_compress_test/dynerf/cook_spinach/point_cloud/iteration_14000/Replaced_point_cloud.ply"
decoded_ply_path = "/home/ubuntu/pc_compress_test/dynerf/cook_spinach/point_cloud/iteration_14000/filtered_point_cloud_decoded.ply"
replace_decoded_ply(input_ply_path, output_filtered_ply_path, decoded_ply_path)

