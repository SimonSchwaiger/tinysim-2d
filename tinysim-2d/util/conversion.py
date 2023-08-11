import copy
import numpy as np
import open3d as o3d

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Cloud utilities
from sensor_msgs_py import point_cloud2 

def euler2quaternion(ai, aj, ak):
    """Creates a quaternion from euler angles
    
    Source: https://docs.ros.org/en/rolling/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Broadcaster-Py.html
    
    Parameters
    ----------
    ai : float
            Pitch of rotation in euler angles
    aj : float
            Yaw of rotation in euler angles
    ak : float
            Roll of rotation in euler angles
                
    Returns
    -------
    array_like
            x, y, z, w of rotation
    """
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.array(
        [
            cj*sc - sj*cs,
            cj*ss + sj*cc,
            cj*cs - sj*sc,
            cj*cc + sj*ss
        ], dtype=float
    )

    return q

## Convert between ROS and O3D pointclouds
# Inspired by but heavily modified: https://github.com/felixchenfy/open3d_ros_pointcloud_conversion/blob/master/lib_cloud_conversion_between_Open3D_and_ROS.py
def pcl2o3dCloud(rosCloud):
    """Converts a ROS PointCloud2 to Open3D pointcloud (with colour)
    """
    # Pull data from ROS cloud. We populate a numpy array of type obj for fancy indexing without casting datatypes
    fieldNames=[ field.name for field in rosCloud.fields ]
    cloudData = np.array(
        list(point_cloud2.read_points(rosCloud, skip_nans=True, field_names = fieldNames)),
        dtype="object"
    )
    # Create o3d cloud object and populate positions (required)
    o3dCloud = o3d.geometry.PointCloud()
    o3dCloud.points = o3d.utility.Vector3dVector(cloudData[:,0:3].astype(float))
    # Optionally add colour value
    idx = np.where(np.array(fieldNames) == "rgba")[0]
    if len(idx) > 0:
        # Unpack from single 4-byte value
        colours = cloudData[:,idx].astype(np.int64).flatten()
        colours = np.array(
            [ struct.unpack("BBBB", struct.pack("I", entry)) for entry in colours ]
        )
        # Convert from RGB to BGR and to range [0,1]
        o3dCloud.colors = o3d.utility.Vector3dVector(
            np.stack(
                (colours[:,2], colours[:,1], colours[:,0]), axis=1
            ).astype(np.float64)/255.0
        )
    # TODO: Add optional normals info
    return o3dCloud

def o3dCloud2pcl(o3dCloud, frame_id="cloud"):
    """Converts an Open3D pointcloud to a ROS PointCloud2 (with colour)
    
    Parameters
    ----------
    o3dCloud : o3d.geometry.PointCloud
            Input point cloud to convert to ROS format
    frame_id : String
            Frame name of the converted point cloud in ROS
                
    Returns
    -------
    PointCloud2
            ROS PointCloud2 containing vertices, triangle and colour data
    
    """
    # Define position fields
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    # Create position points
    points = np.asarray(o3dCloud.points).astype(np.float64)
    points = points.tolist()
    # Add optional rgba component
    if len(np.asarray(o3dCloud.colors)) > 0:
        # Update message fields
        fields.append(PointField('rgba', 12, PointField.UINT32, 1))
        # Change range from [0,1] to [0, 255] and order from rgb to bgra
        colorArr = (np.asarray(o3dCloud.colors)*255).astype(np.int64)
        colorArr  = np.array(
            [ struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0] for (r, g, b) in colorArr ]
        ).astype(np.int64).tolist()
        points =  [ point + [colour] for point, colour in zip(points, colorArr) ]
    # Set frame_id in header and finally create pcl using the convenience function
    header = Header()
    header.frame_id = frame_id
    return point_cloud2.create_cloud(header, fields, points)

def mesh2ROSMeshMarker(vertices, triangles, robotName=""):
    """Creates a ROS visualisation Marker based on given vertices and triangles
    
    Parameters
    ----------
    vertices : array_like
            List of floats (n,3) depicting vertex positions (x, y, z) in world coordinates 
    triangles : array_like
            List of ints (n,3) depicting vertex indices that form polygons 
    robotName : String
            Namespace of the marker's frame_id
                
    Returns
    -------
    Marker
            Visualisation msgs Marker depicting the mesh described by vertices and triangles
    
    """
    def constructPoint(xyz):
        """Unpacks list of coordinates into Point message
        """
        p = Point()
        p.x, p.y, p.z = np.array(xyz, dtype=np.float32).tolist()
        return p
    # Create marker and set to triangle list (i.e. mesh)
    vizMarker = Marker()
    vizMarker.type = 11
    vizMarker.ns = robotName
    vizMarker.frame_locked = True
    # Prevent double backslash in frame name if robotName is empty
    if robotName == "": vizMarker.header.frame_id = "odom"
    else: vizMarker.header.frame_id = "{}/odom".format(robotName)

    vizMarker.scale.x = 1.0
    vizMarker.scale.y = 1.0
    vizMarker.scale.z = 1.0
    
    # ROS uses a vertex' position in the list to determine polygons
    # Therefore, we need to reorder vertices accordingly
    messageVertices = []
    
    for polygonTriangles in triangles:
        messageVertices.append(vertices[polygonTriangles[0]])
        messageVertices.append(vertices[polygonTriangles[1]])
        messageVertices.append(vertices[polygonTriangles[2]])
    
    # Create list of points
    vizMarker.points = [ constructPoint(vertex) for vertex in messageVertices ]
    
    return vizMarker
    
    