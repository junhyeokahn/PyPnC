import pybullet
import rospy
from sensor_msgs.msg import JointState, Image, PointCloud2
from geometry_msgs.msg import Pose, TransformStamped
# from std_msgs.msg import Int16
from util import pybullet_util
from threading import Lock
from simulator.pybullet.rosnode.srv import MoveEndEffectorToSrv, GripperCommandSrv, InterruptSrv, MoveEndEffectorToSrvResponse, GripperCommandSrvResponse, InterruptSrvResponse
# import tf
from scipy.spatial.transform import Rotation
import numpy as np

class DracoManipulationRosnode():
    def __init__(self, robot, link_id):
        rospy.init_node('draco')
        self._robot = robot
        self._link_id = link_id

        # Publishers
        self._joint_pub = rospy.Publisher('draco/joint_pos', JointState, queue_size=1)
        # TODO: what is/where is the ee pose defined?
        # ee_pub = rospy.Publisher('ee_pos', Pose, queue_size=1)
        self._pc_pub = rospy.Publisher('draco/pointcloud', PointCloud2, queue_size=1)
        self._depth_pub = rospy.Publisher('draco/depth', Image, queue_size=1)
        self._image_pub = rospy.Publisher('draco/image_raw', Image, queue_size=1)
        self._camera_transform_pub = rospy.Publisher('draco/camera_transform', TransformStamped, queue_size=1)

        # Services
        # self._command_srv = rospy.Subscriber('draco/command', Int16, self.set_command, queue_size=1)
        # Manually copied GripperCommand message file into this project and changed message constructors in
        #   srv file without changing msg type in desc/headers. Will this work? -> Seems to
        # There were no conda-forge packages for control_msgs even though they existed for geometry, sensor, etc
        self._move_ee_srv = rospy.Service('draco/gripper_command_srv', GripperCommandSrv, self.handle_gripper_command)
        self._move_ee_srv = rospy.Service('draco/end_effector_command_srv', MoveEndEffectorToSrv, self.handle_ee_command)
        self._move_ee_srv = rospy.Service('draco/interrupt_srv', InterruptSrv, self.handle_interrupt)

        # Temp stand in for actual service calls
        self._command_lock = Lock()
        self._remote_command = -1

    def set_command(self, com):
        print("setCommand, have com", com)
        self._command_lock.acquire()
        try:
            self._remote_command = com#.data
        finally:
            self._command_lock.release()

    def get_command(self):
        self._command_lock.acquire()
        try:
            command = self._remote_command
            self._remote_command = -1
        finally:
            self._command_lock.release()
            return command

    # def apply_command(self, interface):
    #     command = -1
    #     self._command_lock.acquire()
    #     try:
    #         if self._remote_command == 8:
    #             interface.interrupt_logic.b_interrupt_button_eight = True
    #         elif self._remote_command == 5:
    #             interface.interrupt_logic.b_interrupt_button_five = True
    #         elif self._remote_command == 4:
    #             interface.interrupt_logic.b_interrupt_button_four = True
    #         elif self._remote_command == 2:
    #             interface.interrupt_logic.b_interrupt_button_two = True
    #         elif self._remote_command == 6:
    #             interface.interrupt_logic.b_interrupt_button_six = True
    #         elif self._remote_command == 7:
    #             interface.interrupt_logic.b_interrupt_button_seven = True
    #         elif self._remote_command == 9:
    #             interface.interrupt_logic.b_interrupt_button_nine = True
    #         elif self._remote_command == 0:
    #             interface.interrupt_logic.b_interrupt_button_zero = True
    #         elif self._remote_command == 1:
    #             interface.interrupt_logic.b_interrupt_button_one = True
    #         elif self._remote_command == 3:
    #             interface.interrupt_logic.b_interrupt_button_three = True
    #         elif self._remote_command == 10: #'t'
    #             interface.interrupt_logic.b_interrupt_button_t = True
    #         elif self._remote_command == 11:
    #             self._remote_command = 'c'
    #         #     for k, v in self._gripper_command.items():
    #         #         self._gripper_command[k] += 1.94 / 3.
    #         elif self._remote_command == 12:
    #             self._remote_command = 'o'
    #         #     for k, v in self._gripper_command.items():
    #         #         self._gripper_command[k] -= 1.94 / 3.
    #         command = self._remote_command
    #         self._remote_command = -1
    #     finally:
    #         self._command_lock.release()
    #         return command

    # TODO: modify these service callbacks to call correspoding scorpiointerface methods once updated
    def handle_gripper_command(self, req):
        print("handle_gripper_command", req)
        self.set_command(11)
        return GripperCommandSrvResponse()

    def handle_ee_command(self, req):
        print("handle_ee_command", req)
        self.set_command(1)
        return MoveEndEffectorToSrvResponse()

    def handle_interrupt(self, req):
        print("handle_interrupt", req)
        self.set_command(3)
        return InterruptSrvResponse()

    def publish_data(self, sensor_data):
        # Publish data to ROS

        #Create ee Pose message
        right_ee_transform = pybullet.getLinkState(self._robot, self._link_id['r_hand_contact'],1,1)
        r_gripper_pose = Pose()
        r_gripper_pose.position.x = right_ee_transform[0][0]
        r_gripper_pose.position.y = right_ee_transform[0][1]
        r_gripper_pose.position.z = right_ee_transform[0][2]
        r_gripper_pose.orientation.x = right_ee_transform[1][0]
        r_gripper_pose.orientation.y = right_ee_transform[1][1]
        r_gripper_pose.orientation.z = right_ee_transform[1][2]
        r_gripper_pose.orientation.w = right_ee_transform[1][3]
        left_ee_transform = pybullet.getLinkState(self._robot, self._link_id['l_hand_contact'],1,1)
        l_gripper_pose = Pose()
        l_gripper_pose.position.x = left_ee_transform[0][0]
        l_gripper_pose.position.y = left_ee_transform[0][1]
        l_gripper_pose.position.z = left_ee_transform[0][2]
        l_gripper_pose.orientation.x = left_ee_transform[1][0]
        l_gripper_pose.orientation.y = left_ee_transform[1][1]
        l_gripper_pose.orientation.z = left_ee_transform[1][2]
        l_gripper_pose.orientation.w = left_ee_transform[1][3]

        # TODO: publish ee poses

        # #Create JointState message
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = sensor_data['joint_pos'].keys()
        joint_msg.position = sensor_data['joint_pos'].values()
        joint_msg.velocity = sensor_data['joint_vel'].values()
        self._joint_pub.publish(joint_msg)

        # Get camera transform
        cam_trans = pybullet.getLinkState(self._robot, self._link_id['camera'],1,1)
        cam_rpy = Rotation.from_quat(cam_trans[1]).as_euler('xyz',degrees=True)

        # Transform for getLinkState and for pybullet not aligned
        cam_rpy[0] -= 90
        # This is simply bc current camera can't actually see enough of the bookshelf
        # Isn't accurate to frame, remove eventually
        cam_rpy[1] -= 15

        camera_transform_msg = TransformStamped()
        camera_transform_msg.transform.translation.x = cam_trans[0][0]
        camera_transform_msg.transform.translation.y = cam_trans[0][1]
        camera_transform_msg.transform.translation.z = cam_trans[0][2]
        camera_quat = Rotation.from_euler('xyz', [cam_rpy[1] - 90, cam_rpy[2], cam_rpy[0]], degrees=True).as_quat()
        camera_transform_msg.transform.rotation.x = camera_quat[0]
        camera_transform_msg.transform.rotation.y = camera_quat[1]
        camera_transform_msg.transform.rotation.z = camera_quat[2]
        camera_transform_msg.transform.rotation.w = camera_quat[3]
        camera_transform_msg.header.frame_id = "camera"
        camera_transform_msg.header.stamp = rospy.Time.now()
        camera_transform_msg.child_frame_id = "world"
        self._camera_transform_pub.publish(camera_transform_msg)

        # Not sure exactly how dist param of pybullet cam data works. Make focus point 1 unit ahead of cam pos and set dist to 1
        cam_pos = np.array(cam_trans[0]) + [1.0,0,0]

        # #Get camera data
        #     depth_msg, image_msg = pybullet_util.get_rgb_and_depth_image(self.camera_transform[0], 1.0, self.camera_transform[1][0], self.camera_transform[1][1], self.camera_transform[1][2], 60., 320, 240, 0.1, 100., 1, 1)
        depth_msg, image_msg = pybullet_util.get_rgb_and_depth_image(cam_pos, 1.0, cam_rpy[0], cam_rpy[1], cam_rpy[2], 60., 320, 240, 0.1, 100., 1, 1)
        self._depth_pub.publish(depth_msg)
        self._image_pub.publish(image_msg)

        # pc_msg, image_msg = pybullet_util.get_xyzrgb_pointcloud2(self.camera_transform[0], 1.0, self.camera_transform[1][0], self.camera_transform[1][1], self.camera_transform[1][2], 60., 320, 240, 0.1, 100., 1, 1)
        # self._pc_pub.publish(pc_msg)
        # self._image_pub.publish(image_msg)