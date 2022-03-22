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

        # Broadcaster
        # pyBullet: x right, y up, z in
        # vision: x right, y down, z out
        # vision rotx is also -Math.PI / 2.0 + rotx (-90 + rotx)
        # Something of the above info seems wrong
        # self.camera_transform = [[1.0,0.0,1.0], [270, 345, 0]]
        self.camera_transform = [[1.0,0.0,1.0], [-90, -15, 0]]
        # self._broadcaster = tf2_ros.StaticTransformBroadcaster()
        # self._broadcaster = tf.TransformBroadcaster()

        # private Matrix4d currVisionTransform = new Matrix4d();    //transform from ground/base(x-right, y-forward, z-up) to vision coordinates (x-right, y-down, z-forward)
        # private double rotX = -Math.PI / 4.0;    //rotation of vision coordinates around x-axis (using vision coordinate system)
        # private double transHeight = 1.5;    //distance of vision sensor from ground (meters)
        # Vision transform with rotx and height:
        #         vision transform: 1.0, 0.0, 0.0, 0.0
                # 0.0, -0.25881904510252063, 0.9659258262890683, 0.0
                # 0.0, -0.9659258262890683, -0.25881904510252063, 1.0
                # 0.0, 0.0, 0.0, 1.0
        #Vision transform with cameraframe transform passed over
                # vision transform: 0.9659258262890682, 0.2588190451025207, -2.7755575615628914E-17, 1.0
                # 0.0, 2.740863092043355E-16, 0.9999999999999999, 0.0
                # 0.25881904510252074, -0.9659258262890681, 2.671474153004283E-16, 1.0
                # 0.0, 0.0, 0.0, 1.0
        self._camera_transform_msg = TransformStamped()
        self._camera_transform_msg.transform.translation.x = self.camera_transform[0][0]
        self._camera_transform_msg.transform.translation.y = self.camera_transform[0][1]
        self._camera_transform_msg.transform.translation.z = self.camera_transform[0][2]
        # camera_quat = Rotation.from_euler('xyz', [self.camera_transform[1][0],self.camera_transform[1][1],self.camera_transform[1][2]], degrees=True).as_quat()
        camera_quat = Rotation.from_euler('xyz', [self.camera_transform[1][1] - 90,self.camera_transform[1][2],self.camera_transform[1][0]], degrees=True).as_quat()
        self._camera_transform_msg.transform.rotation.x = camera_quat[0]
        self._camera_transform_msg.transform.rotation.y = camera_quat[1]
        self._camera_transform_msg.transform.rotation.z = camera_quat[2]
        self._camera_transform_msg.transform.rotation.w = camera_quat[3]
        self._camera_transform_msg.header.frame_id = "camera"
        self._camera_transform_msg.header.stamp = rospy.Time.now()
        self._camera_transform_msg.child_frame_id = "world"

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

    def apply_command(self, interface):
        command = -1
        self._command_lock.acquire()
        try:
            if self._remote_command == 8:
                interface.interrupt_logic.b_interrupt_button_eight = True
            elif self._remote_command == 5:
                interface.interrupt_logic.b_interrupt_button_five = True
            elif self._remote_command == 4:
                interface.interrupt_logic.b_interrupt_button_four = True
            elif self._remote_command == 2:
                interface.interrupt_logic.b_interrupt_button_two = True
            elif self._remote_command == 6:
                interface.interrupt_logic.b_interrupt_button_six = True
            elif self._remote_command == 7:
                interface.interrupt_logic.b_interrupt_button_seven = True
            elif self._remote_command == 9:
                interface.interrupt_logic.b_interrupt_button_nine = True
            elif self._remote_command == 0:
                interface.interrupt_logic.b_interrupt_button_zero = True
            elif self._remote_command == 1:
                interface.interrupt_logic.b_interrupt_button_one = True
            elif self._remote_command == 3:
                interface.interrupt_logic.b_interrupt_button_three = True
            elif self._remote_command == 10:
                interface.interrupt_logic.b_interrupt_button_t = True
            # elif self._remote_command == 11:
            #     for k, v in self._gripper_command.items():
            #         self._gripper_command[k] += 1.94 / 3.
            # elif self._remote_command == 12:
            #     for k, v in self._gripper_command.items():
            #         self._gripper_command[k] -= 1.94 / 3.
            command = self._remote_command
            self._remote_command = -1
        finally:
            self._command_lock.release()
            return command

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
        # right_ee_transform = pybullet_util.get_link_iso(self._robot, self._link_id['right_ezgripper_palm_link'])
        # r_gripper_pose = Pose()
        # r_gripper_pose.position.x = right_ee_transform[0][3]
        # r_gripper_pose.position.y = right_ee_transform[1][3]
        # r_gripper_pose.position.z = right_ee_transform[2][3]
        # r_gripper_quat = Rotation.from_matrix(right_ee_transform[:3,:3]).as_quat()
        # r_gripper_pose.orientation.x = r_gripper_quat[0]
        # r_gripper_pose.orientation.y = r_gripper_quat[1]
        # r_gripper_pose.orientation.z = r_gripper_quat[2]
        # r_gripper_pose.orientation.w = r_gripper_quat[3]
        # left_ee_transform = pybullet_util.get_link_iso(self._robot, self._link_id['left_ezgripper_palm_link'])
        # l_gripper_pose = Pose()
        # l_gripper_pose.position.x = left_ee_transform[0][3]
        # l_gripper_pose.position.y = left_ee_transform[1][3]
        # l_gripper_pose.position.z = left_ee_transform[2][3]
        # l_gripper_quat = Rotation.from_matrix(left_ee_transform[:3,:3]).as_quat()
        # l_gripper_pose.orientation.x = l_gripper_quat[0]
        # l_gripper_pose.orientation.y = l_gripper_quat[1]
        # l_gripper_pose.orientation.z = l_gripper_quat[2]
        # l_gripper_pose.orientation.w = l_gripper_quat[3]
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
        l_gripper_pose.position.x = left_ee_transform[0][1]
        l_gripper_pose.position.y = left_ee_transform[0][2]
        l_gripper_pose.position.z = left_ee_transform[0][3]
        l_gripper_pose.orientation.x = left_ee_transform[1][0]
        l_gripper_pose.orientation.y = left_ee_transform[1][1]
        l_gripper_pose.orientation.z = left_ee_transform[1][2]
        l_gripper_pose.orientation.w = left_ee_transform[1][3]
        pybullet.getLinkState(self._robot, self._link_id['camera'],1,1)

        # #Create JointState message
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = sensor_data['joint_pos'].keys()
        joint_msg.position = sensor_data['joint_pos'].values()
        joint_msg.velocity = sensor_data['joint_vel'].values()
        self._joint_pub.publish(joint_msg)

        # #Get camera data
        depth_msg, image_msg = pybullet_util.get_rgb_and_depth_image(self.camera_transform[0], 1.0, self.camera_transform[1][0], self.camera_transform[1][1], self.camera_transform[1][2], 60., 320, 240, 0.1, 100., 1, 1)
        self._depth_pub.publish(depth_msg)
        self._image_pub.publish(image_msg)

        # pc_msg, image_msg = pybullet_util.get_xyzrgb_pointcloud2(self.camera_transform[0], 1.0, self.camera_transform[1][0], self.camera_transform[1][1], self.camera_transform[1][2], 60., 320, 240, 0.1, 100., 1, 1)
        # self._pc_pub.publish(pc_msg)
        # self._image_pub.publish(image_msg)

        # TODO: how to get updated camera transform when moving?
        self._camera_transform_pub.publish(self._camera_transform_msg)
        # self._broadcaster.sendTransform((1.0, 0.0, 1.0),
        #                                 tf.transformations.quaternion_from_euler(-90, -15, 0),
        #                                 rospy.Time.now(),
        #                                 "camera",
        #                                 "world")