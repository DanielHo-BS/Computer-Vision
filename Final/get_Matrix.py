import numpy as np
import yaml
import xml.etree.ElementTree as ET

def convert_toMatrix(para_list):
    [x, y, z, qx, qy, qz, qw] = para_list
    trans_matrix = np.array([[1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qw*qy + qx*qz), x],
                             [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx), y],
                             [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2), z]])
 
    return trans_matrix

def get_IntrinsicMatrix(cam_info_path):
    cam_direction = ['f', 'fl', 'fr', 'b']
    Int_matrix = {}
    for dir in cam_direction:
        with open(f'{cam_info_path}/gige_100_{dir}_hdr_camera_info.yaml', 'r') as stream:
            data = yaml.load(stream, Loader=yaml.FullLoader)
            cam_intrin = np.array(data['camera_matrix']['data']).reshape((3, 3))
            Int_matrix[dir] = cam_intrin
    
    return Int_matrix


def get_ExtrinsicMatrix(cam_info_path):
    f = ET.parse(f'{cam_info_path}/camera_extrinsic_static_tf.launch')
    root = f.getroot()
    Ext_matrix = {}
    trans_dir = ['f-fr', 'f-fl', 'fl-b', 'no-use', 'base_link-f']

    for idx, node in enumerate(root.iter('node')):
        para = node.attrib['args'].split(' ')[:7]
        para = [float(p) for p in para]
        Ext_matrix[trans_dir[idx]] = convert_toMatrix(para)
    
    return Ext_matrix

    