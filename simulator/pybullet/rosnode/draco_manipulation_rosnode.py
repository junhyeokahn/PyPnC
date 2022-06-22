import time

import pybullet
import rospy
from sensor_msgs.msg import JointState, Image, PointCloud2
from geometry_msgs.msg import Pose, TransformStamped
# from std_msgs.msg import Int16
from util import pybullet_util
from threading import Lock
from simulator.pybullet.rosnode.srv import MoveEndEffectorToSrv, GripperCommandSrv, InterruptSrv, MoveEndEffectorToSrvResponse, GripperCommandSrvResponse, InterruptSrvResponse
from simulator.pybullet.rosnode.srv._LocomotionCommandSrv import LocomotionCommandSrv, LocomotionCommandSrvRequest, LocomotionCommandSrvResponse
from simulator.pybullet.rosnode.srv._ReturnEESrv import ReturnEESrv, ReturnEESrvResponse
from simulator.pybullet.rosnode.srv._SafetyAssessmentSrv import SafetyAssessmentSrv, SafetyAssessmentSrvResponse
# import tf
from scipy.spatial.transform import Rotation
import numpy as np
from util import util
import pickle

t_gripper_stab_dur = 0.1


SAFETY_THRESHOLD = 0.4

grid_location = util.GridLocation(np.array([0.05, 0.05, 0.05]))
saf_list = [None] * 7
for i in range(7):
    with open('saf/saf_{}.pkl'.format(i), 'rb') as f:
        saf_list[i] = pickle.load(f)
# for val in saf_list[6]:
#     if val[2] == 1:
#         print(val, saf_list[6][val])
# quit()


def is_lh_reachable(sensor_data, global_goal):
    """
    0: falling right
    1: falling left
    2: falling forward
    3: falling backward
    4: falling down
    5: falling up
    6: no reaching
    """
    local_goal = global_goal - sensor_data['base_com_pos']
    local_goal[2] = global_goal[2]
    normalized_goal = local_goal - np.array([0.6, 0.25, 0.85])
    grid_idx = grid_location.get_grid_idx(normalized_goal)
    labels = [False] * 7
    print("lhand")
    print("grid: ", grid_idx)
    for i in range(7):
        print("{}th prob: {}".format(i, saf_list[i][grid_idx]))
        labels[i] = False if saf_list[i][grid_idx] < SAFETY_THRESHOLD else True

    return labels


def is_rh_reachable(sensor_data, global_goal):
    """
    0: falling right
    1: falling left
    2: falling forward
    3: falling backward
    4: falling down
    5: falling up
    6: no reaching
    """
    local_goal = global_goal - sensor_data['base_com_pos']
    local_goal[1] *= -1
    local_goal[2] = global_goal[2]
    normalized_goal = local_goal - np.array([0.6, 0.25, 0.85])
    grid_idx = grid_location.get_grid_idx(normalized_goal)
    labels = [False] * 7
    print("rhand")
    print("grid: ", grid_idx)
    for i in range(7):
        print("{}th prob: {}".format(i, saf_list[i][grid_idx]))
        labels[i] = False if saf_list[i][grid_idx] < SAFETY_THRESHOLD else True

    return labels


def is_colliding(start, waypoint, goal, obs_min, obs_max, threshold, n):
    """
    start, waypoint, goal, OBS_MIN, OBS_MAX, THRESHOLD: 3d np.array
    N: number of check point
    """
    if util.is_colliding_3d(start, waypoint, obs_min, obs_max, threshold, n):
        return false
    if util.is_colliding_3d(waypoint, goal, obs_min, obs_max, threshold, n):
        return false
    return true

class DracoManipulationRosnode():
    def __init__(self, robot, link_id, gripper_command):
        rospy.init_node('draco')
        self._robot = robot
        self._link_id = link_id

        # For demo, going to stop doing vision publishing when we are done using it for speed reasons
        self.command_recieved = False

        # Publishers
        self._joint_pub = rospy.Publisher('draco/joint_pos', JointState, queue_size=1)
        self._l_ee_pub = rospy.Publisher('draco/l_ee_pose', Pose, queue_size=1)
        self._r_ee_pub = rospy.Publisher('draco/r_ee_pose', Pose, queue_size=1)
        self._pc_pub = rospy.Publisher('draco/pointcloud', PointCloud2, queue_size=1)
        self._depth_pub = rospy.Publisher('draco/depth', Image, queue_size=1)
        self._image_pub = rospy.Publisher('draco/image_raw', Image, queue_size=1)
        self._camera_transform_pub = rospy.Publisher('draco/camera_transform', TransformStamped, queue_size=1)

        # Services
        # Manually copied GripperCommand message file into this project and changed message constructors in
        #   srv file without changing msg type in desc/headers. Will this work? -> Seems to
        # There were no conda-forge packages for control_msgs even though they existed for geometry, sensor, etc
        self._gripper_srv = rospy.Service('draco/gripper_command_srv', GripperCommandSrv, self.handle_gripper_command)
        self._move_ee_srv = rospy.Service('draco/end_effector_command_srv', MoveEndEffectorToSrv, self.handle_ee_command)
        # Is this still a thing? Walk in place?
        # self._interrupt_srv = rospy.Service('draco/interrupt_srv', InterruptSrv, self.handle_interrupt)
        self._walk_srv = rospy.Service('draco/locomotion_srv', LocomotionCommandSrv, self.handle_locomotion_command)
        self._return_srv = rospy.Service('draco/return_ee_srv', ReturnEESrv, self.handle_return_ee_command)
        self._safety_srv = rospy.Service('draco/safety_assessment', SafetyAssessmentSrv, self.handle_safety_assessment)

        # Current behavior would be for pending commands of the same type to be overwritten if still pending, but
        #   if already active then the newly incoming command would be discarded.
        # TODO: is the ^ what we want?

        # Lock to be held by main apply_command function during each simulation iteration, which should need to be
        #   available for service calls to go through
        self._command_iteration_lock = Lock()

        # Whether each type of command is ready to be recieved [left_gripper,right_gripper,left_hand,right_hand,walk]
        self._ready_state = np.ones(5)
        # Commands:
        #   Adjust left gripper
        #   Adjust right gripper
        #   Move left hand
        #   Move right hand
        #   Walk in X
        #   Walk in Y
        #   Return left ee nominal
        #   Return right ee nominal
        self._command_state = np.zeros(8)

        self._lh_target_pos = np.array([0., 0., 0.])
        self._lh_waypoint_pos = np.array([0., 0., 0.])
        self._lh_target_quat = np.array([0., 0., 0., 1.])
        self._rh_target_pos = np.array([0., 0., 0.])
        self._rh_waypoint_pos = np.array([0., 0., 0.])
        self._rh_target_quat = np.array([0., 0., 0., 1.])
        self._com_target_x = 0.
        self._com_target_y = 0.
        self._gripper_command = gripper_command
        self._t_left_gripper_command_recv = 0
        self._t_right_gripper_command_recv = 0

    def apply_commands(self, interface, t):
        self._command_iteration_lock.acquire()

        # Copy results of any pending commands and return as targets
        gripper_command = self._gripper_command.copy()
        lh_target_pos = self._lh_target_pos.copy()
        lh_waypoint_pos = self._lh_waypoint_pos.copy()
        lh_target_quat = self._lh_target_quat.copy()
        rh_target_pos = self._rh_target_pos.copy()
        rh_waypoint_pos = self._rh_waypoint_pos.copy()
        rh_target_quat = self._rh_target_quat.copy()
        com_displacement_x = self._com_target_x
        com_displacement_y = self._com_target_y

        # Update any relevant interruption interface
        if self._command_state[0]:
            print("got left gripper command: ", gripper_command)
            self._t_left_gripper_command_recv = t
            self._ready_state[0] = 0
            self._command_state[0] = 0
        elif self._command_state[1]:
            print("got right gripper command: ", gripper_command)
            self._t_right_gripper_command_recv = t
            self._ready_state[1] = 0
            self._command_state[1] = 0
        elif self._command_state[2]:
            print("got left ee command: ", lh_target_pos, lh_target_quat)
            interface.interrupt_logic.lh_target_pos = lh_target_pos
            # # lh_target_rot = np.dot(RIGHTUP_GRIPPER, x_rot(-np.pi / 4.))
            # # # lh_target_rot = np.copy(RIGHTUP_GRIPPER)
            # # lh_target_quat = util.rot_to_quat(lh_target_rot)
            # # lh_target_iso = liegroup.RpToTrans(lh_target_rot, lh_target_pos)
            # # lh_waypoint_pos = generate_keypoint(lh_target_iso)[0:3, 3]
            # lh_waypoint_pos = lh_target_pos - [0.1,0.,0.]
            # interface.interrupt_logic.lh_waypoint_pos = lh_waypoint_pos
            interface.interrupt_logic.lh_target_quat = lh_target_quat
            interface.interrupt_logic.b_interrupt_button_one = True
            self._command_state[2] = 0
        elif self._command_state[3]:
            print("got right ee command: ", rh_target_pos, rh_target_quat)
            interface.interrupt_logic.rh_target_pos = rh_target_pos
            # rh_waypoint_pos = rh_target_pos - [0.1,0,0]
            # interface.interrupt_logic.rh_waypoint_pos = rh_waypoint_pos
            interface.interrupt_logic.rh_target_quat = rh_target_quat
            interface.interrupt_logic.b_interrupt_button_three = True
            self._command_state[3] = 0
        elif self._command_state[4] and com_displacement_x != 0:
            print("got walk in x command: ", com_displacement_x)
            interface.interrupt_logic.com_displacement_x = com_displacement_x
            interface.interrupt_logic.b_interrupt_button_m = True
            self._command_state[4] = 0
        elif self._command_state[5] and com_displacement_y != 0:
            print("got walk in y command: ", com_displacement_y)
            interface.interrupt_logic.com_displacement_y = com_displacement_y
            interface.interrupt_logic.b_interrupt_button_n = True
            self._command_state[5] = 0
        # TODO: not sure of behavior in case of conflicting return nominal and move ee commands
        elif self._command_state[6]:
            print("got return left ee to nominal command")
            interface.interrupt_logic.b_interrupt_button_e = True
            self._command_state[6] = 0
        elif self._command_state[7]:
            print("got return right ee to nominal command")
            interface.interrupt_logic.b_interrupt_button_r = True
            self._command_state[7] = 0

        # Update internal standby variables
        if t_gripper_stab_dur < t <= self._t_left_gripper_command_recv + t_gripper_stab_dur:
            self._ready_state[0] = 0
        else:
            self._ready_state[0] = 1
        if t_gripper_stab_dur < t <= self._t_right_gripper_command_recv + t_gripper_stab_dur:
            self._ready_state[1] = 0
        else:
            self._ready_state[1] = 1
        self._ready_state[2] = interface.interrupt_logic.b_left_hand_ready
        self._ready_state[3] = interface.interrupt_logic.b_right_hand_ready
        self._ready_state[4] = interface.interrupt_logic.b_walk_ready

        self._command_iteration_lock.release()
        return lh_target_pos, rh_target_pos, lh_target_quat, rh_target_quat, gripper_command, lh_waypoint_pos, rh_waypoint_pos

    def handle_gripper_command(self, req):
        print("handle_gripper_command", req)

        # Wait for current simulation iteration to finish applying any pending commands and update ready_state, then
        #   process incoming command
        self._command_iteration_lock.acquire()
        resp = GripperCommandSrvResponse()
        side = int(req.side)
        if req.side:
            if self._ready_state[1]:
                for k, v in self._gripper_command.items():
                    if k.split('_')[0] == "right":
                        self._gripper_command[k] = req.command.position
                print("setting right gripper command state to 1")
                self._command_state[1] = 1
            else:
                print("gripper not ready, returning success = false for right gripper command")
                resp.success = False
                self._command_iteration_lock.release()
                return resp
        else:
            if self._ready_state[0]:
                for k, v in self._gripper_command.items():
                    if k.split('_')[0] == "left":
                        self._gripper_command[k] = req.command.position
                print("setting left gripper command state to 1")
                self._command_state[0] = 1
            else:
                print("gripper not ready, returning success = false for left gripper command")
                resp.success = False
                self._command_iteration_lock.release()
                return resp
        self._command_iteration_lock.release()

        # Now results of command will be input next sim iteration, want to wait on command being done (ready_state for
        #   component to be True again)
        print("waiting for gripper state to be ready again")
        while not self._ready_state[side] or self._command_state[side]:
            time.sleep(1)
        print("gripper command finished, returning response")

        # TODO: check if command was actually successful or not by querying gripper location
        resp.success = True

        return resp

    def handle_ee_command(self, req):
        print("handle_ee_command", req)

        # Wait for current simulation iteration to finish applying any pending commands and update ready_state, then
        #   process incoming command
        self._command_iteration_lock.acquire()
        resp = MoveEndEffectorToSrvResponse()

        if req.side:
            if self._ready_state[3]:
                print("right ee command ready, setting")
                self._rh_target_pos = np.array([req.ee_pose.position.x, req.ee_pose.position.y, req.ee_pose.position.z])
                self._rh_target_quat = np.array([req.ee_pose.orientation.x, req.ee_pose.orientation.y, req.ee_pose.orientation.z, req.ee_pose.orientation.w])
                self._command_state[3] = 1
            else:
                print("right ee command not ready, returning false")
                resp.success = False
                self._command_iteration_lock.release()
                return resp
        else:
            if self._ready_state[2]:
                print("left ee command ready, setting")
                self._lh_target_pos = np.array([req.ee_pose.position.x, req.ee_pose.position.y, req.ee_pose.position.z])
                self._lh_target_quat = np.array([req.ee_pose.orientation.x, req.ee_pose.orientation.y, req.ee_pose.orientation.z, req.ee_pose.orientation.w])
                self._command_state[2] = 1
            else:
                print("left ee command not ready, returning false")
                resp.success = False
                self._command_iteration_lock.release()
                return resp
        self._command_iteration_lock.release()

        # Now results of command will be input next sim iteration, want to wait on command being done (ready_state for
        #   component to be True again)
        print("waiting on ee to be ready again")
        while not self._ready_state[2+int(req.side)] or self._command_state[2+int(req.side)]:
            time.sleep(1)
        print("ee ready again, returning")

        # TODO: check if command was actually successful or not by querying ee location
        #   Also, do we still even care about return val if we don't need to manually verify things?
        resp.success = True

        return resp

    def handle_return_ee_command(self, req):
        print("handle_ee_return_command", req)

        # Wait for current simulation iteration to finish applying any pending commands and update ready_state, then
        #   process incoming command
        self._command_iteration_lock.acquire()
        resp = ReturnEESrvResponse()

        if req.side:
            if self._ready_state[3]:
                print("right ee command ready, setting")
                self._command_state[7] = 1
            else:
                print("right ee command not ready, returning false")
                resp.success = False
                self._command_iteration_lock.release()
                return resp
        else:
            if self._ready_state[2]:
                print("left ee command ready, setting")
                self._command_state[6] = 1
            else:
                print("left ee command not ready, returning false")
                resp.success = False
                self._command_iteration_lock.release()
                return resp
        self._command_iteration_lock.release()

        # Now results of command will be input next sim iteration, want to wait on command being done (ready_state for
        #   component to be True again)
        print("waiting on ee to be ready again")
#        Seemingly there is a delay before ready is set to false for this but not ee_reach
        time.sleep(2)
        while not self._ready_state[2+int(req.side)] or self._command_state[6+int(req.side)]:
            time.sleep(1)
        print("ee ready again, returning")

        # TODO: check if command was actually successful or not by querying ee location
        #   Also, do we still even care about return val if we don't need to manually verify things?
        resp.success = True

        return resp

    # # Is this still even a thing?
    # def handle_interrupt(self, req):
    #     print("handle_interrupt", req)
    #     # self.set_command(3)
    #     return InterruptSrvResponse()

    def handle_locomotion_command(self, req):
        print("handle_locomotion_command")

        # Wait for current simulation iteration to finish applying any pending commands and update ready_state, then
        #   process incoming command
        self._command_iteration_lock.acquire()
        resp = LocomotionCommandSrvResponse()
        # TODO: allow separate x and y locomotion commands to not overwrite each other
        if self._ready_state[4]:
            self._com_target_y = req.y
            self._com_target_x = req.x
            self._command_state[4] = 1
            self._command_state[5] = 1
        else:
            resp.success = False
            self._command_iteration_lock.release()
            return resp
        self._command_iteration_lock.release()

        # Now results of command will be input next sim iteration, want to wait on command being done (ready_state for
        #   component to be True again)
        while not self._ready_state[4] or self._command_state[4]:
            time.sleep(1)

        # TODO: check if command was actually successful or not by querying base location
        resp.success = True

        return resp

    def handle_safety_assessment(self, req):
        #call safety assessment function with req.ee_pose and req.side
        if (req.side):
            safety_vals = is_rh_reachable(self.sensor_data, [req.ee_pose.position.x,req.ee_pose.position.y,req.ee_pose.position.z])
        else:
            safety_vals = is_lh_reachable(self.sensor_data,[req.ee_pose.position.x,req.ee_pose.position.y,req.ee_pose.position.z])
#         safety_vals = [False * 7]
        print(safety_vals)

        #create response object
        resp = SafetyAssessmentSrvResponse()
        resp.safetyVals = safety_vals

        return resp

    # Publish data to ROS
    def publish_data(self, sensor_data):
        self.sensor_data = sensor_data

        # Commenting out for demo speed reasons
        #         #Create ee Pose message
        #         right_ee_transform = pybullet.getLinkState(self._robot, self._link_id['r_hand_contact'],1,1)
        #         r_gripper_pose = Pose()
        #         r_gripper_pose.position.x = right_ee_transform[0][0]
        #         r_gripper_pose.position.y = right_ee_transform[0][1]
        #         r_gripper_pose.position.z = right_ee_transform[0][2]
        #         r_gripper_pose.orientation.x = right_ee_transform[1][0]
        #         r_gripper_pose.orientation.y = right_ee_transform[1][1]
        #         r_gripper_pose.orientation.z = right_ee_transform[1][2]
        #         r_gripper_pose.orientation.w = right_ee_transform[1][3]
        #         left_ee_transform = pybullet.getLinkState(self._robot, self._link_id['l_hand_contact'],1,1)
        #         l_gripper_pose = Pose()
        #         l_gripper_pose.position.x = left_ee_transform[0][0]
        #         l_gripper_pose.position.y = left_ee_transform[0][1]
        #         l_gripper_pose.position.z = left_ee_transform[0][2]
        #         l_gripper_pose.orientation.x = left_ee_transform[1][0]
        #         l_gripper_pose.orientation.y = left_ee_transform[1][1]
        #         l_gripper_pose.orientation.z = left_ee_transform[1][2]
        #         l_gripper_pose.orientation.w = left_ee_transform[1][3]
        #
        #         self._l_ee_pub.publish(l_gripper_pose)
        #         self._r_ee_pub.publish(r_gripper_pose)
        #
        #         # Create JointState message
        #         joint_msg = JointState()
        #         joint_msg.header.stamp = rospy.Time.now()
        #         joint_msg.name = sensor_data['joint_pos'].keys()
        #         joint_msg.position = sensor_data['joint_pos'].values()
        #         joint_msg.velocity = sensor_data['joint_vel'].values()
        #         self._joint_pub.publish(joint_msg)


        nearval = 0.1
        ## Build view matrix to pass to pybullet to get rgb and depth buffers

        # ## Use camera frame to build view_matrix and get corresponding sensor data
        # # Get rotation matrix from camera link transform
        # #   cam_trans contains
        # #       link com pos wrt world, link com ori wrt world
        # #       link com pos wrt frame, link com ori wrt frame
        # #       frame pos wrt world, frame ori wrt world
        # #       _, _
        cam_trans = pybullet.getLinkState(self._robot, self._link_id['camera'],1,1)
        cam_pos = np.asarray(cam_trans[0])
        cam_rot = pybullet.getMatrixFromQuaternion(cam_trans[1])

        ## Testing: keeping pos and orientation static
        cam_rpy = np.array([10,20,10])
        cam_pos = [0.0,0.0,0.9]
        # cam_pos = [-0.2,0.4,0.9]
        cam_ori_p = Rotation.from_euler('xyz', cam_rpy, degrees=True).as_quat()
        cam_rot = pybullet.getMatrixFromQuaternion(cam_ori_p)

        # Given Camera transform, compute view matrix to get correct pybullet camera frames
        cam_rot = np.array(cam_rot).reshape(3,3)
        global_camera_x_unit = np.array([1, 0, 0])
        global_camera_z_unit = np.array([0, 0, 1])
        camera_eye_pos = cam_pos
        camera_target_pos = cam_pos + np.dot(cam_rot, 1.0 * global_camera_x_unit)
        camera_up_vector = np.dot(cam_rot, global_camera_z_unit)
        view_matrix = pybullet.computeViewMatrix(camera_eye_pos, camera_target_pos,
                                          camera_up_vector)

        # Using intrinsic parameters and method in vision to get view coords
        # float fx = 155.88456630706787;
        # float fy = 277.12812423706055;
        # float cx = 120;
        # float cy = 160;
        # x_ir = ((xx - cx) / fx) * z_ir;
        # y_ir = ((yy - cy) / fy) * z_ir;
        # >>> ((xx - cx) / fx) * pc
        # -0.16013487932956347
        # >>> ((yy - cy) / fy) * pc
        # -0.09007586875612396
        # 0.2496255616575137

        # Multiplying point by inverse projection matrix to get view coords
        #              z_ir = 2.0 * z_ir - 1.0;
        #              x_ir = ((xx - imgWidth/2.0) / (imgWidth/2.0)) * 0.7698;
        #              y_ir = (((yy - imgHeight/2.0) / (imgHeight/2.0))) * 0.57735;
        #              scale = (-4.995 * z_ir) + 5.005;
        #              x_ir = x_ir / scale;
        #              y_ir = y_ir / scale;
        #              z_ir = -1.0 / scale;
        #
        #              //experimenting with aligning points with visions notion of camera axes
        #              x_ir = x_ir;
        #              y_ir = y_ir;
        #              z_ir = -z_ir;

        # xx 40 yy 100 z 0.6
        # Method 1:
        # -0.102640052255077
        # -0.04330129037845328
        # 0.2
        # Method 2:
        # -0.2875249003984064
        # -0.04792081673306773
        # .2
        # Either intrinsic parameters are incorrect, or there is no 1:1 correspondence from projection matrix


        # view matrix is world to camera
        # Seemingly pybullet base: (x forward, y left, z up)
        # Seemingly pybullet cam: (x right, y up, z back)
        # x becomes neg z
        # y becomes neg x
        # z becomes y
        # [1,2,3] -> [-2,3,-1]
        # array([-3., -1.,  2.])

        # This view matrix assumes axes of
        # [[ 0.  -1.   0.  -0. ]
        #  [-0.   0.   1.  -0.9]
        # [-1.  -0.  -0.   0. ]
        # [ 0.   0.   0.   1. ]]

        # From base to vision
        # view_mat = np.array([x[:3] for x in np.asarray(view_matrix).reshape([4,4],order='F')[:3]])

        # From vision to base
        view_mat = np.linalg.inv(np.asarray(view_matrix).reshape([4,4],order='F'))
        view_mat_rot = np.array([x[:3] for x in view_mat[:3]])
        vision_quat = Rotation.from_matrix(view_mat_rot).as_quat()

        camera_transform_msg = TransformStamped()
        camera_transform_msg.transform.translation.x = view_mat[0][3]#view_matrix[12]#cam_pos[0]#cam_pos[0]
        camera_transform_msg.transform.translation.y = view_mat[1][3]#view_matrix[13]#cam_pos[1]#cam_pos[1]
        camera_transform_msg.transform.translation.z = view_mat[2][3]#view_matrix[14]#cam_pos[2]#cam_pos[2]
        camera_transform_msg.transform.rotation.x = vision_quat[0]
        camera_transform_msg.transform.rotation.y = vision_quat[1]
        camera_transform_msg.transform.rotation.z = vision_quat[2]
        camera_transform_msg.transform.rotation.w = vision_quat[3]
        camera_transform_msg.header.frame_id = "camera"
        camera_transform_msg.header.stamp = rospy.Time.now()
        camera_transform_msg.child_frame_id = "world"
        self._camera_transform_pub.publish(camera_transform_msg)


        # Stopping capture once recieving a command for demo speed reasons
        if np.any(self._command_state):
            self.command_recieved = True

        if not self.command_recieved:
            # pc_msg, image_msg = pybullet_util.get_xyzrgb_pointcloud2(view_matrix, 60., 320, 240, nearval, 100., 1, 1)
            depth_msg, image_msg = pybullet_util.get_rgb_and_depth_image(view_matrix, 60., 320, 240, nearval, 100., 1, 1)
            # self._pc_pub.publish(pc_msg)
            self._depth_pub.publish(depth_msg)
            self._image_pub.publish(image_msg)