import numpy as np


def homogeneous_rotation_matrix(theta) -> np.ndarray:
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return rotation_matrix

def homogeneous_translation_matrix(translation) -> np.ndarray:
    translation_matrix = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])
    return translation_matrix

def homogeneous_transformation_matrix(translation, theta):
    translation_matrix = homogeneous_translation_matrix(translation)
    rotation_matrix = homogeneous_rotation_matrix(theta)
    transformation_matrix = np.dot(translation_matrix, rotation_matrix)
    return transformation_matrix

def decompose_transformation_matrix(T):
    translation = T[:2, 2]
    rotation = T[:2, :2]
    return translation, rotation

def extract_theta_from_rotation(rotation_matrix):
    theta = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return theta

def coordinate_transfer(tf, pose2d):    
    t, r = decompose_transformation_matrix(np.dot(tf, homogeneous_transformation_matrix(pose2d[:-1], pose2d[-1])))
    return np.array([t[0], t[1], extract_theta_from_rotation(r)])


def mapping_period(ori_angle, angle_range) -> np.ndarray:
    lower, upper = angle_range
    period = upper - lower
    res_angle = ori_angle % period
    # res_angle[res_angle > upper] -= period
    if res_angle > upper:
        res_angle = res_angle - period
    return res_angle


def analyze_init_condition(val):
    assert isinstance(val, list), f"'{val}' is not list!"
    if len(val) == 1:
        return val[0]
    elif len(val) == 2:
        return np.random.uniform(*val)
    elif len(val) >= 3:
        return np.random.choice(val)
    else:
        NotImplementedError


if __name__ == '__main__':
    pass