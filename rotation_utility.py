import numpy as np
import kinproc.jtsFunctions
import kinproc.process
import kinproc.rot
import kinproc.utility
import nvtx

@nvtx.annotate(f"()", color = "purple")
def getRotations(sequence, matrix, rangerot2=0):
    # Calculate the corresponding Euler angles from a 3x3 rotation matrix 
    # using the specified sequence.

    # Note that this program follows the convention in the field of robotics: 
    # We use column vectors (not row vectors) to denote vectors/points in 3D 
    # Euclidean space:
    #     V_global = R * V_local
    # where R is a 3x3 rotation matrix, and V_* are 3x1 column vectors. 
    # In this convention, for example, an R_z (rotation about z axis) is 
    # [c -s 0; s c 0; 0 0 1] and NOT [c s 0; -s c 0; 0 0 1], (c = cosine and s = sine)

    # We also acknowledge that Euler angle rotations are always about the axes of 
    # the local coordinate system, never the global coordinate system. (If you would 
    # rather recognize the existence of Euler angles with global coordinate system based
    # rotations and further confuse people in the world, simply reverse the order: an x(3 degrees)-y(4deg)-z(5deg)
    # global-base-rotation is simply a z(5)-y(4)-x(3) local based rotation.) Following the equation
    # and convention above, an x-y-z (1-2-3) Euler angle sequence would mean:
    #     V_global = R * V_local
    #              = R_x * R_y * R_x * V_local
    
    # Sequence:       The rotation sequence used for output (e.g., 312, 213)
    
    # Matrix:         The 3x3 rotation matrix (the so-called Direction Cosine
    #                 Matrix, or DCM). 4x4 homogeneous transformation matrix is acceptable, 
    #                 so long as the first 3 rows and columns are a rotation matrix.

    # rangerot2:      Optional. The range of the second rotation in output. It should be a 
    #                 value of 0 (default) or 1. For symmetric Euler sequences, if rangerot2 == 0,
    #                 the 2nd otation is in the range [0, pi]. If rangerot2 == 1, the range is [-pi,0].
    #                 If the second rotation is 0 or pi, singularity occurs.
                    
    #                 For asymmetric Euler sequences:
    #                 if rangerot== 0, the 2nd rotation is in the range [-pi/2, pi/2]; if rangerot==1, 
    #                 the range is [pi/2, pi*3/2]. If the second rotation is +/- pi/2, singularity occurs.

    # Author:         Shang Mu, 2005-2010
    # Revision:       v8, 2010-05-23. 2010-07-08
    # Revision:       Amiya Gupta, 2021-08-09
    # Python:         2021-08-09

   
    def __c312(M):
        s2 = M[2,1]                 # x rot
        if (rangerot2 == 0):
            rot2 = np.arcsin(s2)
        else:
            rot2 = np.pi-np.arcsin(s2)
        if s2 == any((0,1)):      # singularity
            rot1 = 0
            rot3 = np.arctan2(M[0,2], M[0,0])
        else:
            if rangerot2 == 0:
                rot1 = np.arctan2(-M[0, 1], M[1,1])     # z rot
                rot3 = np.arctan2(-M[2, 0], M[2,2])       # y rot
            else:
                rot1 = np.arctan2(M[0,1], -M[1,1])      # z rot
                rot3 = np.arctan2(M[2,0], -M[2,2])      # y rot
        return rot1, rot2, rot3
  
    def __c313(M):
        c2 = M[2,2]     #x rot
        if rangerot2 == 0:
            rot2 = np.arccos(c2)
        else:
            rot2 = -np.arccos(c2)
        if c2 == any((-1,1)):     # singularity
            rot1 = 0
            rot3 = np.arctan2(M[2,0],M[2,1])
        else:
            if rangerot2 == 0:
                rot1 = np.arctan2(M[0,2], -M[1,2])      # z rot
                rot3 = np.arctan2(M[2,0],M[2,1])        # y rot
            else:
                rot1 = np.arctan2(-M[0,2],M[1,2])       #z rot
                rot3 = np.arctan2(-M[2,0], -M[2,1])     #y rot
        return rot1, rot2, rot3

    __rot120 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    if rangerot2 != 1 and rangerot2 != 0:
        raise Exception('Invalid value for parameter rangerot2')

    M = matrix[0:3, 0:3]
    sequence = "c" + str(sequence)

    # Asymmetric Sequences
    if sequence == "c312":
        [rot1, rot2, rot3] = __c312(M)
    elif sequence == "c123":
        [rot1, rot2, rot3] = __c312(__rot120.T @ M @ __rot120)
    elif sequence == "c213":
        [rot1, rot2, rot3] = np.negative(np.flip(__c312(M.T)))
    elif sequence == "c231":
        M = __rot120.T @ M @ __rot120
        [rot1, rot2, rot3] = __c312(__rot120.T @ M @ __rot120)
    elif sequence == "c321":
        [rot1, rot2, rot3] = np.negative(np.flip(__c312(__rot120.T @ M @ __rot120)))
    elif sequence == "c132":
        M = __rot120.T @ M @ __rot120 #231
        [rot1, rot2, rot3] = np.negative(np.flip(__c312(__rot120.T @ M @ __rot120)))

    # Symmetric Sequences
    elif sequence == "c313":
        [rot1, rot2, rot3] = __c313(M)
    elif sequence == "c212":
        [rot1, rot2, rot3] = __c313(rot.x(-90).T @ M @ rot.x(-90))
    elif sequence == "c232":
        [rot1, rot2, rot3] = __c313(__rot120 @ M @ __rot120.T)
    elif sequence == "c131":
        M =__rot120 @ M @ __rot120.T
        [rot1, rot2, rot3] = __c313(rot.x(-90).T @ M @ rot.x(-90))
    elif sequence == "c121":
        M = __rot120 @ M @ __rot120.T
        M = rot.x(-90).T @ M @ rot.x(-90)
        [rot1, rot2, rot3] = __c313(rot.x(-90).T @ M @ rot.x(-90))

    else:
        raise Exception('getRotations(): Sequence not yet supported')
        rotations = []
    
    return np.rad2deg(rot1), np.rad2deg(rot2), np.rad2deg(rot3)
    # Sidenote:

    # These methods are used to compute the sequences from 312 and 313 to simplify the program and increase the efficiency.

    # There are many ways to simplify this program:

    # Preliminary (for methods 1 and 2):
    # The 12 different Euler angle sequences can be divided into 4 groups:
    # (123, 231, 312), (321, 213, 132) , (121, 232, 313), (323, 212, 131).
    # (The first two groups can be further combined to a single one using
    # method 3 below. Similarly the last two groups can be united by method 4.) 

    # Simplification method 1:
    # Basic idea: if I have two different coordinate systems on the same rigid
    # body, an x-rotation seen from one coordinate system can be a y-rotation
    # in the other coordinate system.
    # The tool is a rotation matrix:
    #   rot120 = [
    #      0     0     1
    #      1     0     0
    #      0     1     0
    #   ];  (a shift of axis indices, or a 120 degree rotation about [1 1 1]).
    # For any sequence, the three Euler angles can be easily calculated using
    # the same program for any other sequence in the same group (as defined in
    # the "preliminary" section). For example (assuming M is 3x3):
    # getRotations(123, M) == getRotations(312, rot120.'*M*rot120)
    #                      == getRotations(231, rot120*M*rot120.');
    # getRotations(232, M) == getRotations(121, rot120.'*M*rot120)
    #                      == getRotations(313, rot120*M*rot120.').

    # Simplification method 2:
    # There are rules we can follow. For example, s2 or c2 (sine or cosine of
    # the 2nd rotation) is always at the (1st rot)(3rd rot) element in the
    # rotation matrix. The four groups we saw above are the only variations we
    # need to take care of.

    # Simplification method 3:
    # An A-B-C order Euler angle rotations of a body EE with respect to some
    # fixed body FF, can be seen as a C-B-A order rotations of the body FF with 
    # respect to the body EE (but with the negative angles). For example
    # (assuming M is 3x3):
    # getRotations(123, M) == -getRotations(321, M.')(end:-1:1);
    # getRotations(312, M) == -getRotations(213, M.')(end:-1:1).

    # Simplification method 4:
    # Similar to method 1, a relabel of axes utilizing 90 degree rotations is
    # extremely useful for the symmetrical sequences. A 90 degree rotation
    # about either of the two axes in a symmetrical sequence would result in a
    # new valid sequence. For example:
    # getRotations(212, M) == getRotations(232, roty(90).'*M*roty(90))
    #                      == getRotations(313, rotx(-90).'*M*rotx(-90)).
    # This essentially unites all the symmetric sequences.
    # This method could also be used on the asymmetric sequences, but care
    # must be taken to negate the sign of individual angles.





# The ambiguous pose function - determines the offset 

@nvtx.annotate(f"()", color = "purple")
def ambiguous_pose(pose):
    xr = kinproc.rot.x(pose[4])
    yr = kinproc.rot.y(pose[5])
    zr = kinproc.rot.z(pose[3])

    R = np.matmul(np.matmul(zr,xr),yr)

    y_ax = R[:,1]
    #print(y_ax)

    com = np.array(pose[0:3])
    normed = com/np.linalg.norm(com)

    amg_by_90 = abs(np.dot(y_ax,normed))
    angle_between = np.rad2deg(np.arccos(amg_by_90))

    return abs(angle_between - 90)



@nvtx.annotate(f"()", color = "purple")
def ambiguous_pose_x(pose):
    xr = kinproc.rot.x(pose[4])
    yr = kinproc.rot.y(pose[5])
    zr = kinproc.rot.z(pose[3])

    R = np.matmul(np.matmul(zr,xr),yr)

    x_ax = R[:,0]
    #print(y_ax)

    com = np.array(pose[0:3])
    normed = com/np.linalg.norm(com)

    amg_by_90 = abs(np.dot(x_ax,normed))
    angle_between = np.rad2deg(np.arccos(amg_by_90))

    return abs(angle_between - 90)


@nvtx.annotate(f"()", color = "purple")
def axis_angle_rotation_matrix(axis, angle):
    m = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    v = 1-c

    return np.array([
        [m[0]*m[0]*v + c, m[0]*m[1]*v - m[2]*s, m[0]*m[2]*v + m[1]*s],
        [m[0]*m[1]*v + m[2]*s, m[1]*m[1]*v + c, m[1]*m[2]*v - m[0]*s],
        [m[0]*m[2]*v - m[1]*s, m[1]*m[2]*v + m[0]*s, m[2]*m[2]*v + c]
    ])



@nvtx.annotate(f"()", color = "purple")
def sym_trap_dual(pose):
    # First, we find the Rotation matrix that describes the pose in space.
    xr = kinproc.rot.x(pose[4])
    yr = kinproc.rot.y(pose[5])
    zr = kinproc.rot.z(pose[3])

    R = np.matmul(np.matmul(zr,xr),yr)
   # print("Rotation Matrix: ", R)
    # We want to look at the z-axis
    z_ax = R[:,2]
    print("Z-axis: ", z_ax)

    # Compare the object's z-axis with the angle of the observer - > center of mass
    com = np.array(pose[0:3])
    #print("Center of mass: ", com)
    normed = com/np.linalg.norm(com)

    # Transform the new vector so that it goes from COM -> observer
    normed = -normed
    print("Normed: ",normed)

    # Find the angle between the two, take abs() to get total angle (we let cross product handle the direction of rotation)
    costh = abs(np.dot(z_ax, normed))
    print("costh: ",costh)
    angle_between = np.rad2deg(np.arccos(costh))
   #
    # Find the axis of rotation, M is notation from Crane and Duffy
    M = np.cross(z_ax, normed)

    # Need to normalize M
    M_norm = M/np.linalg.norm(M)
    print("M: ",M_norm)

    # At this point, we know that the amount we want to rotate is double the angle between z_ax and V
    desired_rotation = 2*angle_between
    des_rot_rad = np.deg2rad(desired_rotation)
    print("Angle Between: ", angle_between)
    print("Desired Rotation: ", desired_rotation)

    # Solve for the rotation matrix (above formula)
    sym_R = axis_angle_rotation_matrix(axis = M_norm, angle = des_rot_rad)
    #print("rotation matrix: ", sym_R)
    # need to tinker with this a little bit
    new_pose = np.matmul(sym_R, R)
    zr,xr,yr = getRotations(sequence="312", matrix=new_pose)

    return pose[0], pose[1], pose[2],  zr ,xr, yr


def sym_trap_solid_distance(pose):
    
    # First, we find the Rotation matrix that describes the pose in space.
    xr = kinproc.rot.x(pose[4])
    yr = kinproc.rot.y(pose[5])
    zr = kinproc.rot.z(pose[3])

    R = np.matmul(np.matmul(zr,xr),yr)
   # print("Rotation Matrix: ", R)
    # We want to look at the z-axis
    z_ax = R[:,2]
    #print("Z-axis: ", z_ax)

    # Compare the object's z-axis with the angle of the observer - > center of mass
    com = np.array(pose[0:3])
    #print("Center of mass: ", com)
    normed = com/np.linalg.norm(com)

    # Transform the new vector so that it goes from COM -> observer
    normed = -normed
    #print("Normed: ",normed)

    # Find the angle between the two, take abs() to get total angle (we let cross product handle the direction of rotation)
    costh = abs(np.dot(z_ax, normed))
    angle_between = np.rad2deg(np.arccos(costh))
   #print("Angle Between: ", angle_between)
    # Find the axis of rotation, M is notation from Crane and Duffy
    M = np.cross(z_ax, normed)

    # Need to normalize M
    M_norm = M/np.linalg.norm(M)
    print("M: ",M_norm)

    # At this point, we know that the amount we want to rotate is double the angle between z_ax and V
    desired_rotation = 2*angle_between

    return desired_rotation


def two_vector_rotation_matrix(vector1,vector2):
    
    v1 = vector1 / np.linalg.norm(vector1)
    v2 = vector2 / np.linalg.norm(vector2)
    #print(v1)
    #print(v2)
    ang = -np.arccos(np.dot(v1,v2)) # in radians
    #print(ang * 180/np.pi)
    m = np.cross(v1,v2) / np.linalg.norm(np.cross(v1,v2))
    
    return axis_angle_rotation_matrix(axis = m, angle = ang)


def create_rotation_matrix_312(zrot,xrot,yrot):
    xr = kinproc.rot.x(xrot)
    yr = kinproc.rot.y(yrot)
    zr = kinproc.rot.z(zrot)
    
    return np.matmul(np.matmul(zr,xr),yr)