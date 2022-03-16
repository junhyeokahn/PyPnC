import rospy
from sensor_msgs.msg import JointState, Image
# from geometry_msgs.msg import Pose, TransformStamped
# from std_msgs.msg import Int16
from util import pybullet_util
from threading import Lock
from simulator.pybullet.rosnode.srv import MoveEndEffectorToSrv, GripperCommandSrv, InterruptSrv, MoveEndEffectorToSrvResponse, GripperCommandSrvResponse, InterruptSrvResponse

class DracoManipulationRosnode():
    def __init__(self):
        rospy.init_node('draco')

        # Publishers
        self._joint_pub = rospy.Publisher('draco/joint_pos', JointState, queue_size=1)
        # ee_pub = rospy.Publisher('ee_pos', Pose, queue_size=1)
        # self._pc_pub = rospy.Publisher('draco/pointcloud', PointCloud2, queue_size=1)
        self._depth_pub = rospy.Publisher('draco/depth', Image, queue_size=1)
        self._image_pub = rospy.Publisher('draco/image_raw', Image, queue_size=1)

        # Services
        # self._command_srv = rospy.Subscriber('draco/command', Int16, self.set_command, queue_size=1)
        # Manually copied GripperCommand message file into this project and changed message constructors in
        #   srv file without changing msg type in desc/headers. Will this work? -> Seems to
        # There were no conda-forge packages for control_msgs even though they existed for geometry, sensor, etc
        self._move_ee_srv = rospy.Service('draco/gripper_command_srv', GripperCommandSrv, self.handle_gripper_command)
        self._move_ee_srv = rospy.Service('draco/end_effector_command_srv', MoveEndEffectorToSrv, self.handle_ee_command)
        self._move_ee_srv = rospy.Service('draco/interrupt_srv', InterruptSrv, self.handle_interrupt)

        # Broadcaster
        # self._broadcaster = tf2_ros.StaticTransformBroadcaster()

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

        # #Create JointState message
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = sensor_data['joint_pos'].keys()
        joint_msg.position = sensor_data['joint_pos'].values()
        joint_msg.velocity = sensor_data['joint_vel'].values()
        self._joint_pub.publish(joint_msg)

        # #Get camera data
        # pc_msg, image_msg = pybullet_util.get_xyzrgb_pointcloud2([1., 0.5, 1.], 1.0, 120, -15, 0, 60., 320, 240, 0.1, 100., 1, 1)
        depth_msg, image_msg = pybullet_util.get_rgb_and_depth_image([1.0,0.0,1.0], 1.0, -90, -15, 0, 60., 320, 240, 0.1, 100., 1, 1)
        self._depth_pub.publish(depth_msg)
        self._image_pub.publish(image_msg)
