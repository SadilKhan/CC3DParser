import math
import numpy as np


def rads_to_degs(rads):
    """Convert an angle from radians to degrees"""
    return 180 * rads / math.pi


def angle_from_vector_to_x(vec):
    """computer the angle (0~2pi) between a unit vector and positive x axis"""
    angle = 0.0
    # 2 | 1
    # -------
    # 3 | 4
    if vec[0] >= 0:
        if vec[1] >= 0:
            # Qadrant 1
            angle = math.asin(vec[1])
        else:
            # Qadrant 4
            angle = 2.0 * math.pi - math.asin(-vec[1])
    else:
        if vec[1] >= 0:
            # Qadrant 2
            angle = math.pi - math.asin(vec[1])
        else:
            # Qadrant 3
            angle = math.pi + math.asin(-vec[1])
    return angle


def cartesian2polar(vec, with_radius=False):
    """convert a vector in cartesian coordinates to polar(spherical) coordinates"""
    #vec = vec.round(6)
    norm = np.linalg.norm(vec)
    if norm==0:
        raise ZeroDivisionError("Norm is 0")
    theta = np.arccos(vec[2] / norm)  # (0, pi)
    # (-pi, pi) # FIXME: -0.0 cannot be identified here
    phi = np.arctan(vec[1] / (vec[0] + 1e-15))
    if vec[1]<0 and vec[0]<0:
        phi=phi-np.pi
    elif vec[0]<0:
        phi=np.pi-phi
    elif vec[1]<0:
        phi*=-1
    if not with_radius:
        return np.array([theta, phi])
    else:
        return np.array([theta, phi, norm])


def polar2cartesian(vec):
    """convert a vector in polar(spherical) coordinates to cartesian coordinates"""
    r = 1 if len(vec) == 2 else vec[2]
    theta, phi = vec[0], vec[1]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def rotate_by_x(vec, theta):
    mat = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    return np.dot(mat, vec)


def rotate_by_y(vec, theta):
    mat = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(mat, vec)


def rotate_by_z(vec, phi):
    mat = np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1]])
    return np.dot(mat, vec)


def polar_parameterization(normal_3d, x_axis_3d):
    """represent a coordinate system by its rotation from the standard 3D coordinate system

    Args:
        normal_3d (np.array): unit vector for normal direction (z-axis)
        x_axis_3d (np.array): unit vector for x-axis

    Returns:
        theta, phi, gamma: axis-angle rotation 
    """
    normal_polar = cartesian2polar(normal_3d)
    theta = normal_polar[0]
    phi = normal_polar[1]

    ref_x = rotate_by_z(rotate_by_y(np.array([1, 0, 0]), theta), phi)

    gamma = np.arccos(np.dot(x_axis_3d, ref_x).round(6))
    if np.dot(np.cross(ref_x, x_axis_3d), normal_3d) < 0:
        gamma = -gamma
    return theta, phi, gamma


def polar_parameterization_inverse(theta, phi, gamma):
    """build a coordinate system by the given rotation from the standard 3D coordinate system"""
    normal_3d = polar2cartesian([theta, phi])
    ref_x = rotate_by_z(rotate_by_y(np.array([1, 0, 0]), theta), phi)
    ref_y = np.cross(normal_3d, ref_x)
    x_axis_3d = ref_x * np.cos(gamma) + ref_y * np.sin(gamma)
    return normal_3d, x_axis_3d


def dot_product(X, Y):
    return np.dot(X, Y)/(np.linalg.norm(X)*np.linalg.norm(Y))


def cross_product(X, Y):
    return np.cross(X, Y)


def rotation_matrix_plane(normal_source, normal_target):
    """
    Returns a rotation matrix to rotate a plane
    """

    costheta = dot_product(normal_source, normal_target)
    axis = cross_product(normal_source, normal_target)
    sintheta = np.sqrt(1-costheta**2)
    C = 1-costheta
    x = axis[0]
    y = axis[1]
    z = axis[2]

    return np.array([[x*x*C+costheta, x*y*C-z*sintheta, x*z*C+y*sintheta],
                     [y*x*C+z*sintheta, y*y*C+costheta, y*z*C-x*sintheta],
                     [z*x*C-y*sintheta, z*y*C+x*sintheta, z*z*C+costheta]])

def point_transformation(points,x_axis=np.array([1,0,0]),y_axis=np.array([0,1,0]), 
                         z_axis=np.array([0,0,1]),origin=np.array([0,0,0]),iftranslation=False):
    """Linear Transformation of points"""
    if len(points.shape)==1:
        points=points.reshape(1,-1)
    
    if points.shape[-1]==2:
        # Add z coordinates
        points=add_axis(points,value=0)
    
    transformed_pts=points[:,0].reshape(-1,1)*x_axis+points[:,1].reshape(-1,1)*y_axis+\
        points[:,2].reshape(-1,1)*z_axis
    if iftranslation:
        transformed_pts+=origin
    return transformed_pts

def add_axis(point,value):
    N=point.shape[0]
    axisValue=np.ones((N,1))*value
    return np.concatenate([point,axisValue],axis=1)


def get_transformation_matrix(x_axis_3d,y_axis_3d,z_axis_3d,translation,iftranslation=False):
    """ Transformation matrix for transforming points to another coordinate system"""
    rotation_mat=np.vstack([x_axis_3d, y_axis_3d,z_axis_3d])
    if iftranslation:
        translation_mat=translation.reshape(-1,1)
    else:
        translation_mat=np.zeros_like(translation.reshape(-1,1))
    transformation_mat=np.concatenate([rotation_mat,translation_mat],axis=1)
    ones=np.array([0,0,0,1]).reshape(1,4)
    translation_mat_homogeneous=np.concatenate([transformation_mat,ones])

    return translation_mat_homogeneous


def euclidean_distance(X, Y):
    """
    X: numpy array
    Y: numpy array
    """
    return np.sqrt(np.sum((X-Y)**2))

def check_distance(X,Y,eps=1e-6):
    assert euclidean_distance(X,Y)<eps, f"Expected same arrays but got two different arrays\n X:{X}\nY:{Y}"

def unit_vector(X):
    """
    Vector normalization
    """
    if np.linalg.norm(X)==1:
        return X
    else:
        return X/np.linalg.norm(X)

def l1_distance(X,Y):
    return np.abs(np.sum(X-Y))
