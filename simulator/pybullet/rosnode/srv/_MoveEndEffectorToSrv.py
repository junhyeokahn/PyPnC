# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from draco_msg_gen/MoveEndEffectorToSrvRequest.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import geometry_msgs.msg

class MoveEndEffectorToSrvRequest(genpy.Message):
  _md5sum = "7970c19e1de920e0f5dc8adb189d5ef3"
  _type = "draco_msg_gen/MoveEndEffectorToSrvRequest"
  _has_header = False  # flag to mark the presence of a Header object
  _full_text = """geometry_msgs/Pose ee_pose
bool side

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
"""
  __slots__ = ['ee_pose','side']
  _slot_types = ['geometry_msgs/Pose','bool']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       ee_pose,side

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(MoveEndEffectorToSrvRequest, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.ee_pose is None:
        self.ee_pose = geometry_msgs.msg.Pose()
      if self.side is None:
        self.side = False
    else:
      self.ee_pose = geometry_msgs.msg.Pose()
      self.side = False

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_get_struct_7dB().pack(_x.ee_pose.position.x, _x.ee_pose.position.y, _x.ee_pose.position.z, _x.ee_pose.orientation.x, _x.ee_pose.orientation.y, _x.ee_pose.orientation.z, _x.ee_pose.orientation.w, _x.side))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.ee_pose is None:
        self.ee_pose = geometry_msgs.msg.Pose()
      end = 0
      _x = self
      start = end
      end += 57
      (_x.ee_pose.position.x, _x.ee_pose.position.y, _x.ee_pose.position.z, _x.ee_pose.orientation.x, _x.ee_pose.orientation.y, _x.ee_pose.orientation.z, _x.ee_pose.orientation.w, _x.side,) = _get_struct_7dB().unpack(str[start:end])
      self.side = bool(self.side)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_get_struct_7dB().pack(_x.ee_pose.position.x, _x.ee_pose.position.y, _x.ee_pose.position.z, _x.ee_pose.orientation.x, _x.ee_pose.orientation.y, _x.ee_pose.orientation.z, _x.ee_pose.orientation.w, _x.side))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.ee_pose is None:
        self.ee_pose = geometry_msgs.msg.Pose()
      end = 0
      _x = self
      start = end
      end += 57
      (_x.ee_pose.position.x, _x.ee_pose.position.y, _x.ee_pose.position.z, _x.ee_pose.orientation.x, _x.ee_pose.orientation.y, _x.ee_pose.orientation.z, _x.ee_pose.orientation.w, _x.side,) = _get_struct_7dB().unpack(str[start:end])
      self.side = bool(self.side)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_7dB = None
def _get_struct_7dB():
    global _struct_7dB
    if _struct_7dB is None:
        _struct_7dB = struct.Struct("<7dB")
    return _struct_7dB
# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from draco_msg_gen/MoveEndEffectorToSrvResponse.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import geometry_msgs.msg

class MoveEndEffectorToSrvResponse(genpy.Message):
  _md5sum = "2f3261f2e9af31d0bdd1c53aa6fb73c1"
  _type = "draco_msg_gen/MoveEndEffectorToSrvResponse"
  _has_header = False  # flag to mark the presence of a Header object
  _full_text = """bool success
geometry_msgs/Pose ee_pose
#int8 success

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
"""
  __slots__ = ['success','ee_pose']
  _slot_types = ['bool','geometry_msgs/Pose']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       success,ee_pose

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(MoveEndEffectorToSrvResponse, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.success is None:
        self.success = False
      if self.ee_pose is None:
        self.ee_pose = geometry_msgs.msg.Pose()
    else:
      self.success = False
      self.ee_pose = geometry_msgs.msg.Pose()

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_get_struct_B7d().pack(_x.success, _x.ee_pose.position.x, _x.ee_pose.position.y, _x.ee_pose.position.z, _x.ee_pose.orientation.x, _x.ee_pose.orientation.y, _x.ee_pose.orientation.z, _x.ee_pose.orientation.w))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.ee_pose is None:
        self.ee_pose = geometry_msgs.msg.Pose()
      end = 0
      _x = self
      start = end
      end += 57
      (_x.success, _x.ee_pose.position.x, _x.ee_pose.position.y, _x.ee_pose.position.z, _x.ee_pose.orientation.x, _x.ee_pose.orientation.y, _x.ee_pose.orientation.z, _x.ee_pose.orientation.w,) = _get_struct_B7d().unpack(str[start:end])
      self.success = bool(self.success)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_get_struct_B7d().pack(_x.success, _x.ee_pose.position.x, _x.ee_pose.position.y, _x.ee_pose.position.z, _x.ee_pose.orientation.x, _x.ee_pose.orientation.y, _x.ee_pose.orientation.z, _x.ee_pose.orientation.w))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.ee_pose is None:
        self.ee_pose = geometry_msgs.msg.Pose()
      end = 0
      _x = self
      start = end
      end += 57
      (_x.success, _x.ee_pose.position.x, _x.ee_pose.position.y, _x.ee_pose.position.z, _x.ee_pose.orientation.x, _x.ee_pose.orientation.y, _x.ee_pose.orientation.z, _x.ee_pose.orientation.w,) = _get_struct_B7d().unpack(str[start:end])
      self.success = bool(self.success)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_B7d = None
def _get_struct_B7d():
    global _struct_B7d
    if _struct_B7d is None:
        _struct_B7d = struct.Struct("<B7d")
    return _struct_B7d
class MoveEndEffectorToSrv(object):
  _type          = 'draco_msg_gen/MoveEndEffectorToSrv'
  _md5sum = '67baf42e203386bd688bcf803715c540'
  _request_class  = MoveEndEffectorToSrvRequest
  _response_class = MoveEndEffectorToSrvResponse
