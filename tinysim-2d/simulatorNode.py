import rclpy
from tf2_ros import TransformBroadcaster

from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker

import numpy as np
import open3d as o3d

from util import mapLoader, conversion

def movementUpdate(robot, pose, raycastingScene, cmd_vel):
    """Performs robot movement based on received cmd_vel messages.
    
    Parameters
    ----------
    robot : {"dt"}
            Contains robot and laserscanner information
    newPose : array_like
            Contains current robot pose (X, Y, theta)
    raycastingScene: o3d.t.geometry.RaycastingScene
            Contains 3D scene used for collision check and simulated scans
    cmv_vel : Twist
            Twist message containing received cmd_vel
            
    Returns
    -------
    array_like
            X, Y, Theta of new pose
    """
    # Create placeholder for new pose
    newPose = np.array(pose, dtype=np.float32)
    if cmd_vel != None:
        ## Apply motion model
        # TODO: more accurate motion model with error
        # dx = cos(theta') * x' * dt
        # dy = sin(theta') * x' * dt
        # dtheta = theta' * dt
        newPose += np.array(
            [
                np.cos(newPose[2]) * cmd_vel.linear.x * robot["dt"],
                np.sin(newPose[2]) * cmd_vel.linear.x * robot["dt"],
                cmd_vel.angular.z * robot["dt"]
            ],
            dtype=np.float32
        )
    ## Check for collision with environment and prevent update in case of collision
    #print("pose: {}, newPose: {}".format(self.pose, newPose))

    #xyMovement = np.array(newPose[:2], dtype=np.float32) - np.array(self.pose[:2], dtype=np.float32)
    #print("xyMovement: {}".format(xyMovement))

    # TODO: Prevent the robot from moving through walls
    # Cast a ray from current pose to the new pose
    collisionRay = np.expand_dims(
        np.hstack(
            (
                pose[:2],
                np.full(1, robot["z_height"]),
                (np.array(newPose[:2]) - np.array(pose[:2])),
                np.full(1, 0)
            )
        ),
        axis=0
    )
    rayTensor = o3d.core.Tensor(collisionRay, dtype = o3d.core.Dtype.Float32)
    raycastResult = raycastingScene.cast_rays(rayTensor)
    # Only apply movement update if distance to hit geometry greater than the distance between the two poses
    travelDistance = np.linalg.norm(np.array(newPose[0] - pose[0]))
    #print("raycastResult: {}, travelDistance: {}".format(raycastResult["t_hit"].numpy().ravel(), travelDistance))
    # TODO: figure out why changing update rate affects how close to a wall the robto gets when colliding
    if raycastResult["t_hit"].numpy().ravel() - travelDistance < 1.0: 
        newPose = np.array(pose, dtype=np.float32)
    
    return newPose

def scannerUpdate(robot, pose, raycastingScene, noiseStd=0.003):
    """Determines current scan using raycasting
    
    Parameters
    ----------
    robot : {"name", "num_scans", "z_height", "angle_min", "angle_max", "dt"}
            Contains robot and laserscanner information
    pose : array_like
            Contains 2D robot pose (x, y, theta)
    raycastingScene: o3d.t.geometry.RaycastingScene
            Contains 3D scene used for collision check and simulated scans
            
    Returns
    -------
    sensor_msgs.msg.LaserScan
            LaserScan message resulting from scene and robot pose
    """
    # Create simulated laserscan using open3d raycasts
    # TODO: add rotation to mount the scanner in different poses onto the robot
    # TODO: skip update if robot has not moved enough
    
    angle_min = 0
    angle_max = 2*np.pi
    num_scans = 360
    
    assert robot["angle_max"] > robot["angle_min"]
    increment = (robot["angle_max"] - robot["angle_min"])/robot["num_scans"]

    # Raycast tensor format: (originx, originy, originz, dirx, diry, dirz)
    # For 2D scanner: (posex, posey, scannerZHeight, dirx, diry, 0)
    rayList = [ 
        np.array(
            [
                pose[0],
                pose[1],
                robot["z_height"],
                np.cos(angle+pose[2]),
                np.sin(angle+pose[2]),
                0
            ],
            dtype = np.float32
        ) for angle
        in np.arange(robot["angle_min"], robot["angle_max"], increment)
    ]

    # Create o3d tensor and perform all raycasts in one go
    rays = np.array(rayList, dtype=np.float32)
    rayTensor = o3d.core.Tensor(rays, dtype = o3d.core.Dtype.Float32)
    raycastResult = raycastingScene.cast_rays(rayTensor)
    # Add gaussian noise using numpy
    noise = np.random.normal(0.0, noiseStd, size=len(raycastResult["t_hit"]))
    
    # Format laserscan message
    scan = LaserScan()
    scan.header.frame_id = "{}/base_link".format(robot["name"])
    scan.angle_min = robot["angle_min"]
    scan.angle_max = robot["angle_max"]
    scan.angle_increment = increment
    scan.time_increment = 0.0
    scan.scan_time = robot["dt"]
    scan.range_min = 0.01
    scan.range_max = 7.0
    scan.ranges = (raycastResult["t_hit"].numpy() + noise).tolist()
    
    return scan

def pose2transform(robot, pose):
    """Converts 2D pose to TransformStamped

    Parameters
    ----------
    robot : {"name"}
            Contains robot and laserscanner information
    pose : array_like
            Contains 2D pose (x, y, theta)

    Returns
    -------
    transform: TransformStamped()
            Transform resulting from pose
    """
    # Create message and fill in time and position
    t = TransformStamped()
    t.header.frame_id = "{}/odom".format(robot["name"])
    t.child_frame_id = "{}/base_link".format(robot["name"])
    t.transform.translation.x = float(pose[0])
    t.transform.translation.y = float(pose[1])
    t.transform.translation.z = 0.0

    # Convert rotation around z to quaternion
    q = conversion.euler2quaternion(0.0, 0.0, pose[2])
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    
    return t

class simNode(rclpy.node.Node):
    """Node containing simulation scene, raycasting scene and ROS 2 connections
    """
    def __init__(self, robot, mapVertices, mapTriangles):
        """Class constructor. Sets up raycasting scene and ros connections
        
        Parameters
        ----------
        robot : {"name", "initial_pose", "angle_min", "angle_max", "num_scans", "z_height", "dt" }
                Contains robot and laserscanner information       
        mapVertices: array_like
                (n,3) np.float32 array containing vertex points in world coordinates for the raycasting scene
        mabTriangles: array_like
                (n,3) np.uint32 array containing triangle indices that create the polygons from mapVertices
        """
        super().__init__("{}_sim_scene".format(robot["name"]))
        
        # Robot config
        self.pose = robot["initial_pose"]
        self.robot = robot
        
        # Raycasting scene and geometry
        self.raycastingScene = o3d.t.geometry.RaycastingScene()
        # Add mesh to scene. We don't store the mesh id because we don't care what the raycast actually hit
        meshID = self.raycastingScene.add_triangles(o3d.core.Tensor(mapVertices), o3d.core.Tensor(mapTriangles))
        
        # Visualise mesh in ROS as mesh marker
        self.meshPub = self.create_publisher(Marker, "{}/mesh".format(robot["name"]), 1)
        self.marker = conversion.formatROSMeshMarker(mapVertices, mapTriangles, robot["name"])
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.meshPub.publish(self.marker)
        
        # Placeholders for subscribers/publishers
        self.tfPub = TransformBroadcaster(self)
        self.scanPub = self.create_publisher(LaserScan, "{}/scan".format(robot["name"]), 1)
        
        self.latestCommand = None
        self.commandSub = self.create_subscription(Twist, "{}/cmd_vel".format(robot["name"]), self.commandCallback, 1)
        
        self.timer = self.create_timer(robot["dt"], self.timerCallback)
        
        
    def commandCallback(self, data):
        """Stores latest cmd_vel
        
        Parameters
        ----------
        data : LaserScan
                Latest message received on cmd_vel topic
        """
        self.latestCommand = data
        
    def timerCallback(self):
        """Timer callback performing time-discrete simulation updates
        """
        # Perform movement update
        self.pose = movementUpdate(self.robot, self.pose, self.raycastingScene, self.latestCommand)
        stamp = self.get_clock().now().to_msg()
              
        # Publish transform
        transform = pose2transform(self.robot, self.pose)
        transform.header.stamp = stamp
        self.tfPub.sendTransform(transform)

        # Perform scan update
        scan = scannerUpdate(self.robot, self.pose, self.raycastingScene)
        scan.header.stamp = stamp
        self.scanPub.publish(scan)
        
if __name__=="__main__":
    # Create robot
    robot = {
        "name": "turtle1",
        "initial_pose": [0,0,0],
        "angle_min": 0.0,
        "angle_max": 2*np.pi,
        "num_scans": 360,
        "z_height": 0.2,
        "dt": 0.01,
        "map_file": "/app/map.yaml",
        "point_cloud": None,
    }
    # Load geometry the simulation is based on
    mapVertices, mapTriangles = mapLoader.reconstruct2dMap(robot["map_file"])
    #TODO: Check if map_file or point_cloud is set and load map/pointcloud accordingly
    #TODO: Pull config from ROS parameter server
    #TODO: json 2 rosparam
    rclpy.init()
    node = simNode(robot, mapVertices, mapTriangles)
    try:
        node.get_logger().info('Beginning client, shut down with CTRL-C')
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')
        node.destroy_node()
        rclpy.shutdown()
    except rclpy.executors.ExternalShutdownException:
        # Only print because ROS' logging will be shut down already
        print('Keyboard interrupt, shutting down.')
    
    