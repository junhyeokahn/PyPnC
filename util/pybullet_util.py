from collections import OrderedDict

import pybullet as p
import numpy as np

from util import util
from util import liegroup


def get_robot_config(robot,
                     initial_pos=None,
                     initial_quat=None,
                     b_print_info=False):
    nq, nv, na, joint_id, link_id = 0, 0, 0, OrderedDict(), OrderedDict()
    link_id[(p.getBodyInfo(robot)[0]).decode("utf-8")] = -1
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        if info[2] != p.JOINT_FIXED:
            joint_id[info[1].decode("utf-8")] = info[0]
        link_id[info[12].decode("utf-8")] = info[0]
        nq = max(nq, info[3])
        nv = max(nv, info[4])
    nq += 1
    nv += 1
    na = len(joint_id)

    base_pos, base_quat = p.getBasePositionAndOrientation(robot)
    rot_world_com = util.quat_to_rot(base_quat)
    initial_pos = [0., 0., 0.] if initial_pos is None else initial_pos
    initial_quat = [0., 0., 0., 1.] if initial_quat is None else initial_quat
    rot_world_basejoint = util.quat_to_rot(np.array(initial_quat))
    pos_basejoint_to_basecom = base_pos - np.array(initial_pos)
    rot_basejoint_to_basecom = np.dot(rot_world_basejoint.transpose(),
                                      rot_world_com)

    if b_print_info:
        print("=" * 80)
        print("SimulationRobot")
        print("nq: ", nq, ", nv: ", nv, ", na: ", na)
        print("Vector from base joint frame to base com frame")
        print(pos_basejoint_to_basecom)
        print("Rotation from base joint frame to base com frame")
        print(rot_basejoint_to_basecom)
        print("+" * 80)
        print("Joint Infos")
        util.pretty_print(joint_id)
        print("+" * 80)
        print("Link Infos")
        util.pretty_print(link_id)

    return nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom


def get_kinematics_config(robot, joint_id, link_id, open_chain_joints,
                          base_link, ee_link):
    joint_screws_in_ee = np.zeros((6, len(open_chain_joints)))
    ee_link_state = p.getLinkState(robot, link_id[ee_link])
    if link_id[base_link] == -1:
        base_pos, base_quat = p.getBasePositionAndOrientation(robot)
    else:
        base_link_state = p.getLinkState(robot, link_id[base_link])
        base_pos, base_quat = base_link_state[0], base_link_state[1]
    T_w_b = RpToTrans(quat_to_rot(np.array(base_quat)), np.array(base_pos))
    T_w_ee = RpToTrans(quat_to_rot(np.array(ee_link_state[1])),
                       np.array(ee_link_state[0]))
    T_b_ee = np.dot(TransInv(T_w_b), T_w_ee)
    for i, joint_name in enumerate(open_chain_joints):
        joint_info = p.getJointInfo(robot, joint_id[joint_name])
        link_name = joint_info[12].decode("utf-8")
        joint_type = joint_info[2]
        joint_axis = joint_info[13]
        screw_at_joint = np.zeros(6)
        link_state = p.getLinkState(robot, link_id[link_name])
        T_w_j = RpToTrans(quat_to_rot(np.array(link_state[5])),
                          np.array(link_state[4]))
        T_ee_j = np.dot(TransInv(T_w_ee), T_w_j)
        Adj_ee_j = Adjoint(T_ee_j)
        if joint_type == p.JOINT_REVOLUTE:
            screw_at_joint[0:3] = np.array(joint_axis)
        elif joint_type == p.JOINT_PRISMATIC:
            screw_at_joint[3:6] = np.array(joint_axis)
        else:
            raise ValueError
        joint_screws_in_ee[:, i] = np.dot(Adj_ee_j, screw_at_joint)

    return joint_screws_in_ee, T_b_ee


def set_joint_friction(robot, joint_id, max_force=0):
    p.setJointMotorControlArray(robot, [*joint_id.values()],
                                p.VELOCITY_CONTROL,
                                forces=[max_force] * len(joint_id))


def set_motor_trq(robot, joint_id, command):
    assert len(joint_id) == len(command['joint_trq'])
    trq_applied = OrderedDict()
    for (joint_name, pos_des), (_, vel_des), (_, trq_des) in zip(
            command['joint_pos'].items(), command['joint_vel'].items(),
            command['joint_trq'].items()):
        joint_state = p.getJointState(robot, joint_id[joint_name])
        joint_pos, joint_vel = joint_state[0], joint_state[1]
        trq_applied[joint_id[joint_name]] = trq_des

    p.setJointMotorControlArray(robot,
                                trq_applied.keys(),
                                controlMode=p.TORQUE_CONTROL,
                                forces=trq_applied.values())


def get_sensor_data(robot, joint_id, link_id, pos_basejoint_to_basecom,
                    rot_basejoint_to_basecom):
    """
    Parameters
    ----------
    joint_id (dict):
        Joint ID Dict
    link_id (dict):
        Link ID Dict
    pos_basejoint_to_basecom (np.ndarray):
        3d vector from base joint frame to base com frame
    rot_basejoint_to_basecom (np.ndarray):
        SO(3) from base joint frame to base com frame
    b_fixed_Base (bool);
        Whether the robot is floating or fixed
    Returns
    -------
    sensor_data (dict):
        base_com_pos (np.array):
            base com pos in world
        base_com_quat (np.array):
            base com quat in world
        base_com_lin_vel (np.array):
            base com lin vel in world
        base_com_ang_vel (np.array):
            base com ang vel in world
        base_joint_pos (np.array):
            base pos in world
        base_joint_quat (np.array):
            base quat in world
        base_joint_lin_vel (np.array):
            base lin vel in world
        base_joint_ang_vel (np.array):
            base ang vel in world
        joint_pos (dict):
            Joint pos
        joint_vel (dict):
            Joint vel
        b_rf_contact (bool):
            Right Foot Contact Switch
        b_lf_contact (bool):
            Left Foot Contact Switch
    """
    sensor_data = OrderedDict()

    # Handle Base Frame Quantities
    base_com_pos, base_com_quat = p.getBasePositionAndOrientation(robot)
    sensor_data['base_com_pos'] = np.asarray(base_com_pos)
    sensor_data['base_com_quat'] = np.asarray(base_com_quat)

    base_com_lin_vel, base_com_ang_vel = p.getBaseVelocity(robot)
    sensor_data['base_com_lin_vel'] = np.asarray(base_com_lin_vel)
    sensor_data['base_com_ang_vel'] = np.asarray(base_com_ang_vel)

    rot_world_com = util.quat_to_rot(np.copy(sensor_data['base_com_quat']))
    rot_world_joint = np.dot(rot_world_com,
                             rot_basejoint_to_basecom.transpose())
    sensor_data['base_joint_pos'] = sensor_data['base_com_pos'] - np.dot(
        rot_world_joint, pos_basejoint_to_basecom)
    sensor_data['base_joint_quat'] = util.rot_to_quat(rot_world_joint)
    trans_joint_com = liegroup.RpToTrans(rot_basejoint_to_basecom,
                                         pos_basejoint_to_basecom)
    adT_joint_com = liegroup.Adjoint(trans_joint_com)
    twist_com_in_world = np.zeros(6)
    twist_com_in_world[0:3] = np.copy(sensor_data['base_com_ang_vel'])
    twist_com_in_world[3:6] = np.copy(sensor_data['base_com_lin_vel'])
    augrot_com_world = np.zeros((6, 6))
    augrot_com_world[0:3, 0:3] = rot_world_com.transpose()
    augrot_com_world[3:6, 3:6] = rot_world_com.transpose()
    twist_com_in_com = np.dot(augrot_com_world, twist_com_in_world)
    twist_joint_in_joint = np.dot(adT_joint_com, twist_com_in_com)
    rot_world_joint = np.dot(rot_world_com,
                             rot_basejoint_to_basecom.transpose())
    augrot_world_joint = np.zeros((6, 6))
    augrot_world_joint[0:3, 0:3] = rot_world_joint
    augrot_world_joint[3:6, 3:6] = rot_world_joint
    twist_joint_in_world = np.dot(augrot_world_joint, twist_joint_in_joint)
    sensor_data['base_joint_lin_vel'] = np.copy(twist_joint_in_world[3:6])
    sensor_data['base_joint_ang_vel'] = np.copy(twist_joint_in_world[0:3])

    # Joint Quantities
    sensor_data['joint_pos'] = OrderedDict()
    sensor_data['joint_vel'] = OrderedDict()
    for k, v in joint_id.items():
        js = p.getJointState(robot, v)
        sensor_data['joint_pos'][k] = js[0]
        sensor_data['joint_vel'][k] = js[1]

    return sensor_data


def get_camera_image(robot, link_id, projection_matrix):
    link_info = p.getLinkState(robot, link_id['head'])  #Get head link info
    link_pos = link_info[0]  #Get link com pos wrt world
    link_ori = link_info[1]  #Get link com ori wrt world
    rot = p.getMatrixFromQuaternion(link_ori)
    rot = np.array(rot).reshape(3, 3)

    global_camera_x_unit = np.array([1, 0, 0])
    global_camera_z_unit = np.array([0, 0, 1])

    camera_eye_pos = link_pos + np.dot(rot, 0.1 * global_camera_x_unit)
    camera_target_pos = link_pos + np.dot(rot, 1.0 * global_camera_x_unit)
    camera_up_vector = np.dot(rot, global_camera_z_unit)
    view_matrix = p.computeViewMatrix(camera_eye_pos, camera_target_pos,
                                      camera_up_vector)
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        50,  #image width
        10,  #image height
        view_matrix,
        projection_matrix)
    return width, height, rgb_img, depth_img, seg_img


def is_key_triggered(keys, key):
    o = ord(key)
    if o in keys:
        return keys[ord(key)] & p.KEY_WAS_TRIGGERED
    return False
