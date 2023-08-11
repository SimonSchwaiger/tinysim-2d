import rclpy
from tf2_ros import TransformBroadcaster

from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import LaserScan, PointCloud2
from visualization_msgs.msg import Marker

import numpy as np
import numpy.typing as npt
import open3d as o3d

import copy
from typing import List, Type, Optional
from typing_extensions import TypedDict

from util import mapLoader, conversion

SensorType = TypedDict("Sensor", {
    "dtype": str, "topic": str, "frame": str, "noiseStd": float,
    "pose": npt.ArrayLike, "angle_min_horizontal": float,
    "angle_max_horizontal": float, "num_scans_horizontal": int,
    "angle_min_vertical": float, "angle_max_vertical": float,
    "num_scans_vertical": int
})

RobotType = TypedDict("Robot", {
    "name": str, "initial_pose": npt.ArrayLike,
    "robot_base_radius": float, "dt": float, "map_file": str,
    "pcd_file": str, "collision_sensor": int, "sensors": List[SensorType]
})

# Testrobot
robot = {
    "name": "",
    "initial_pose": [0, 0, 0],
    "robot_base_radius": 0.1,
    "dt": 0.05,
    "map_file": "/app/map.yaml",
    "pcd_file": "",
    "collision_sensor": 0,
    "sensors": [
        {
            "dType": "LaserScan", # Turtlebot 2D Scanner
            "topic": "scan",
            "frame": "scan_front",
            "noiseStd": 0.005,
            "pose": [0.1, 0, 0.2, 0, 0, 0], # [x, y, z, r, p, y]
            "angle_min_horizontal": 0.0,
            "angle_max_horizontal": 2*np.pi,
            "num_scans_horizontal": 360,
            "angle_min_vertical": 0,
            "angle_max_vertical": 0,
            "num_scans_vertical": 1,
        },
        {
            "dType": "PointCloud2", # Velodyne Puck
            "topic": "velodyne_points",
            "frame": "scan_top",
            "noiseStd": 0.0005,
            "pose": [-0.1, 0, 0.4, 0, np.pi/16, 0], # [x, y, z, r, p, y]
            "angle_min_horizontal": 0.0,
            "angle_max_horizontal": 2*np.pi,
            "num_scans_horizontal": 180,
            "angle_min_vertical": -15*np.pi/180,
            "angle_max_vertical": 15*np.pi/180,
            "num_scans_vertical": 16,
        }
    ]
}

class rayCreator:
    def __init__(self, sensorConfig: SensorType) -> None:
        self.sensor: SensorType = None # Sensor configuration
        self.mesh = None   # O3d triangleMesh containing ray directions (no typehint because of https://github.com/isl-org/Open3D/issues/3052 )
        self.loadConfig(sensorConfig)
    
    def loadConfig(self, sensorConfig: SensorType) -> None:
        """Stores an (updated) sensor config and caches config-dependent raycast directions
        """
        self.sensor = sensorConfig
        ## Create correctly oriented ray directions and store them so we don't have to compute them each update
        # Create list of required horizontal and vertical angles per sensor plane
        hAngles = self.createAngles(
            self.sensor["angle_min_horizontal"], self.sensor["angle_max_horizontal"], self.sensor["num_scans_horizontal"]
        )
        vAngles = self.createAngles(
            self.sensor["angle_min_vertical"], self.sensor["angle_max_vertical"], self.sensor["num_scans_vertical"]
        )
        # Placeholder for direction portion of raycast
        castDirections = []
        # Iterate over vertical angles and create sensor planes
        for vAngle in vAngles:
            castDirections.append([
                [ np.cos(hAngle), np.sin(hAngle), np.sin(vAngle) ] 
                for hAngle in hAngles
            ])

        # Create o3d mesh object containing rays and rotate according to sensor orientation
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(np.array(castDirections).reshape([-1, 3]))
        self.mesh.rotate(self.mesh.get_rotation_matrix_from_xyz(self.sensor["pose"][3:6]))
        
    def computeRays(self, robotPose: npt.ArrayLike) -> np.ndarray:
        """Calculates the scan's rays based on stored sensor configuration and current robot pose
        """
        # Copy stored mesh and rotate it according to robot rotation
        tmpMesh = copy.deepcopy(self.mesh)
        tmpMesh.rotate(self.mesh.get_rotation_matrix_from_xyz([0, 0, robotPose[2]]))
        castDirections = np.asarray(tmpMesh.vertices)
        # Translate the rotated mesh
        robotPosition = np.array([robotPose[0], robotPose[1], 0], dtype=float)
        # Since the sensor position is relative to the robot base, we need to rotate the sensor accordingly
        sensorPosition = robotPosition + np.array([
            np.cos(robotPose[2])*self.sensor["pose"][0] + np.sin(robotPose[2])*self.sensor["pose"][1], #TODO: test y translation
            np.sin(robotPose[2])*self.sensor["pose"][0] + np.cos(robotPose[2])*self.sensor["pose"][1],
            self.sensor["pose"][2]
        ], dtype=float)
        #sensorPosition = robotPosition + np.array(self.sensor["pose"][:3], dtype=float)
        return np.hstack((np.full(castDirections.shape, sensorPosition), castDirections))
        
    @staticmethod
    def createAngles(minAngle: float, maxAngle: float, numScans: int) -> np.ndarray:
        """Creates an array of length numScans from minAngle to maxAngle.
        """
        # Handle single angle (i.e. 2D scanner)
        if numScans <= 1: return [(maxAngle - minAngle)/2]
        # Create the array
        increment = (maxAngle - minAngle)/numScans
        return np.array(
            [ angle for angle in np.arange(minAngle, maxAngle, increment) ]
        )

class sensorContainer:
    def __init__(self, robotConfig: RobotType, mapVertices: npt.ArrayLike, mapTriangles: npt.ArrayLike) -> None:
        self.robot: RobotType = None # Sensor configuration
        self.sensors: Optional[List[Type[rayCreator]]] = None
        self.raycastingScene = None
        self.mesh = None
        
        self.loadConfig(robotConfig)
        self.loadGeometry(mapVertices, mapTriangles)
    
    def loadConfig(self, robotConfig: RobotType) -> None:
        """Stores an (updated) robot config and caches config-dependent members
        """
        #TODO: instead of re-instantiating every sensor, check what has changed in the config and use sensor.loadConfig
        self.robot = robotConfig
        ## Instantiate rayCreators
        self.sensors = [ rayCreator(sensor) for sensor in self.robot["sensors"] ]
    
    def loadGeometry(self, mapVertices: npt.ArrayLike, mapTriangles: npt.ArrayLike) -> None:
        """(Re)loads a mesh (consisting of vertices and triangles in o3d notation) and creates a raycasting scene
        """
        mesh = o3d.t.geometry.TriangleMesh(o3d.core.Device("CPU:0"))
        mesh.vertex.positions = o3d.core.Tensor(mapVertices)
        mesh.triangle.indices = o3d.core.Tensor(mapTriangles)
        self.raycastingScene = o3d.t.geometry.RaycastingScene()
        self.raycastingScene.add_triangles(mesh)
        self.mesh = mesh
        
    def getGeometry(self):
        """Getter for the mesh stored in self.raycastingScene
        """
        return self.mesh.fill_holes() #TODO fix artefacts that happen when filling mesh holes
        
    def update(self, robotPose: npt.ArrayLike):
        """Simulates one sensor update based on current robot pose
        """
        # Update Sensor poses
        rayList = [ sensor.computeRays(robotPose) for sensor in self.sensors]
        sections = np.cumsum([ len(entry) for entry in rayList ])
        rayListFlat = np.concatenate(rayList).astype(np.float32)
        # Call raycast
        rayTensor = o3d.core.Tensor(rayListFlat, dtype = o3d.core.Dtype.Float32)
        raycastResult = self.raycastingScene.cast_rays(rayTensor)
        tHitSections = np.split(raycastResult["t_hit"].numpy(), sections)
        # Return resulting rays and hit distances
        return rayList, tHitSections

def formatROSMsg(sensor: SensorType, resultData):
    """Performs sensor-specific updates and formats corresponding ROS 2 message
    """
    def LaserScanMessage(sensor: SensorType, resultData):
        """2D LaserScan requires resultData["t_hit"]
        """
        scan = LaserScan()
        scan.header.frame_id = sensor["frame"]
        scan.angle_min = sensor["angle_min_horizontal"]
        scan.angle_max = sensor["angle_max_horizontal"]
        scan.angle_increment = (sensor["angle_max_horizontal"] - sensor["angle_min_horizontal"])/sensor["num_scans_horizontal"]
        scan.time_increment = 0.0
        scan.scan_time = 0.001
        scan.range_min = 0.01
        scan.range_max = 7.0
        scan.ranges = (resultData["t_hit"]).tolist()
        return scan
    
    def PointCloud2Message(sensor: SensorType, resultData):
        """3D scan requires resultData["t_hit"] and resultData["rays"]
        """
        # Convert ray length to point in 3d space
        hit = np.isfinite(resultData['t_hit'])
        points = resultData["rays"][hit][:,:3] + resultData["rays"][hit][:,3:]*resultData['t_hit'][hit].reshape((-1,1))
        # Track points in o3d point cloud and convert to ros point cloud
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        return conversion.o3dCloud2pcl(pcd, frame_id="odom")          #sensor["frame"])
    
    switchTypes = {
        "LaserScan": LaserScanMessage,
        "PointCloud2": PointCloud2Message
    }
    try:
        msg = switchTypes[sensor["dType"]](sensor, resultData)
    except KeyError:
        msg = None
    
    return msg

def formatROSTransform(frame: str, childFrame: str, pose: npt.ArrayLike, stamp=None):
    """
    """
    t = TransformStamped()
    t.header.frame_id = frame
    t.child_frame_id = childFrame
    t.transform.translation.x = float(pose[0])
    t.transform.translation.y = float(pose[1])
    t.transform.translation.z = float(pose[2])
    q = conversion.euler2quaternion(*pose[3:])
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    if stamp is not None: transform.header.stamp = stamp
    return t

def movementUpdate(pose: npt.ArrayLike, cmd_vel, dt: float):
    """Performs robot movement based on received cmd_vel messages. 
    Method is separate from simScene to make inclusion of external physics simulators possible
    """
    # Create placeholder for new pose
    newPose = np.array(pose, dtype=np.float32)
    travelDistance = rotationDistance = 0
    if cmd_vel != None:
        ## Apply linear motion model without error
        newPose += np.array(
            [
                np.cos(newPose[2]) * cmd_vel.linear.x * dt,
                np.sin(newPose[2]) * cmd_vel.linear.x * dt,
                cmd_vel.angular.z * dt
            ],
            dtype=np.float32
        )
        # Track travelled distance
        travelDistance = abs(cmd_vel.linear.x*dt)
        rotationDistance = abs(cmd_vel.angular.z*dt)
    
    return newPose, travelDistance, rotationDistance

class simulationScene:
    def __init__(self, robotConfig: RobotType) -> None:
        self.robot: RobotType = robotConfig
        self.robotPose: npt.ArrayLike = robot["initial_pose"]
        # Create raycasting scene and instantiate sensor container 
        mapVertices, mapTriangles = mapLoader.reconstruct2DMap(self.robot["map_file"])
        self.sensors: Optional[Type[sensorContainer]] = sensorContainer(robot, mapVertices, mapTriangles)
        
        # Store static transforms
        self.poses = [ formatROSTransform("base_link", sensor["frame"], sensor["pose"]) for sensor in self.robot["sensors"] ]
        
        self.cachedRayList: Optional[npt.ArrayLike] = None
        self.cachedtHitSections: Optional[npt.ArrayLike] = None
        
    def loadConfig(self, robotConfig: RobotType) -> None:
        pass
        # Determine difference between configs: https://stackoverflow.com/questions/32815640/how-to-get-the-difference-between-two-dictionaries-in-python
        
    def update(self, cmd_vel):
        """Performs a discrete simulation update based on the provided command velocity (will hopefully change a lot to be cleaner)
        """
        # Apply motion and sensor updates
        newPose, travelDistance, rotationDistance = movementUpdate(self.robotPose, cmd_vel, robot["dt"])
        
        # Only perform scan update if robot has moved
        if travelDistance > 0 or rotationDistance > 0 or self.cachedRayList is None:
            newRayList, newtHitSections = self.sensors.update(newPose)
        
            # Only apply update permanently if robot is not colliding
            #TODO find a better, actual solution in movement update
            if np.amin(newtHitSections[robot["collision_sensor"]]) > self.robot["robot_base_radius"]:
                self.robotPose = newPose
                self.cachedRayList = newRayList
                self.cachedtHitSections = newtHitSections
        
        # Create scanner messages and add gaussian noise to simulate sensor error
        pubMessages = [
            formatROSMsg(sensor, {
                "t_hit": self.cachedtHitSections[i] + np.random.normal(
                    0.0, sensor["noiseStd"], size=len(self.cachedtHitSections[i])
                ),
                "rays": self.cachedRayList[i]
            })
            for i, sensor in enumerate(self.robot["sensors"])
        ]
        
        # Return all resulting transformations and messages
        retTransforms = self.poses + [formatROSTransform("odom", "base_link", [self.robotPose[0], self.robotPose[1], 0, 0, 0, self.robotPose[2]])]   
        return retTransforms, pubMessages
        
class simNode(rclpy.node.Node):
    def __init__(self, robotConfig: RobotType) -> None:
        super().__init__("{}_sim_scene".format(robot["name"]))
        self.simScene = simulationScene(robot)
        
        # Sensor and transform publishers
        self.tfPub = TransformBroadcaster(self)
        self.sensorPubs = [
            self.create_publisher(eval(sensor["dType"]), sensor["topic"], 1)
            for sensor in robotConfig["sensors"]
        ]
        
        # Mesh visualisation publisher
        self.meshPub = self.create_publisher(Marker, "{}/mesh".format(robotConfig["name"]), 1)
        newMesh = self.simScene.sensors.getGeometry()
        # TODO: change mesh2ROSMeshMarker to take o3d mesh directly
        self.marker = conversion.mesh2ROSMeshMarker(newMesh.vertex.positions.numpy(), newMesh.triangle.indices.numpy(), robot["name"])
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.meshPub.publish(self.marker)
        
        # Command subscriber
        self.latestCommand = None
        self.commandSub = self.create_subscription(Twist, "{}/cmd_vel".format(robot["name"]), self.commandCallback, 1)
        
        # Update timer
        self.timer = self.create_timer(robot["dt"], self.timerCallback)
        
    def commandCallback(self, data) -> None:
        """Stores latest cmd_vel
        
        Parameters
        ----------
        data : LaserScan
                Latest message received on cmd_vel topic
        """
        self.latestCommand = data
        
    def timerCallback(self) -> None:
        """Timer callback performing time-discrete simulation updates
        """
        stamp = self.get_clock().now().to_msg()
        transforms, messages = self.simScene.update(self.latestCommand)
        
        for t in transforms:
            t.header.stamp = stamp
            self.tfPub.sendTransform(t)
        
        for idx, m in enumerate(messages):
            m.header.stamp = stamp
            self.sensorPubs[idx].publish(m)
            
        self.meshPub.publish(self.marker)
        
if __name__=="__main__":
    #TODO: Type Checking of ROS 2 Types: https://discourse.ros.org/t/python-type-checking-in-ros2/28507/3
    #TODO: Check if map_file is yaml or point_cloud and load map/pointcloud accordingly
    #TODO: Pull config from ROS parameter server: https://docs.ros.org/en/rolling/Tutorials/Beginner-Client-Libraries/Using-Parameters-In-A-Class-Python.html and http://design.ros2.org/articles/ros_parameters.html
    #TODO: json 2 rosparam
    #TODO: adjust architecture to make other sensor types possible (depth camera & imu); also explore what a rasteriser with open3d would look like
    rclpy.init()
    node = simNode(robot)
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
