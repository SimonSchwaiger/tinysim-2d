import rclpy
from tf2_ros import TransformBroadcaster

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import LaserScan, PointCloud2, Image
from visualization_msgs.msg import Marker

import cv2
from cv_bridge import CvBridge

import numpy as np
import numpy.typing as npt
import open3d as o3d

import copy
from typing import List, Tuple, Type, Optional, Union
from typing_extensions import TypedDict

from util import mapLoader, conversion

ArrayListType = Union[List, npt.NDArray]

SensorTypeLaserScan = TypedDict("SensorTypeLaserScan", {
    "dType": str, "topic": str, "frame": str, "noiseStd": float,
    "pose": ArrayListType, "angle_min_horizontal": float,
    "angle_max_horizontal": float, "num_scans_horizontal": int,
    "angle_min_vertical": float, "angle_max_vertical": float,
    "num_scans_vertical": int
})

SensorTypeDepthCamera = TypedDict("SensorTypeDepthCamera", {
    "dType": str, "topic": str, "frame": str, "noiseStd": float,
    "pose": ArrayListType, "resolution_horizontal": int,
    "resolution_vertical": int, "fov": int, "scaling_factor": int
})

SensorType = Union[SensorTypeLaserScan, SensorTypeDepthCamera]

RobotType = TypedDict("RobotType", {
    "name": str, "initial_pose": ArrayListType,
    "robot_base_radius": float, "dt": float, "map_file": str,
    "pcd_file": str, "collision_sensor": int, 
    "sensors": List[SensorType]
})

RayCastResultType = TypedDict("RayCastResultType", {"t_hit": ArrayListType, "rays": ArrayListType})

# Testrobot
robot: RobotType = {
    "name": "",
    "initial_pose": [0, 0, 0],
    "robot_base_radius": 0.1,
    "dt": 0.1,
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
            "dType": "PointCloud2", # Velodyne Puck (VLP16)
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
        },
        {
            "dType": "Image", # Realsense d435i
            "topic": "camera/depth/compressed",
            "frame": "camera",
            "pose": [0, 0, 0.5, 0, 0, 0],
            "resolution_horizontal": 640,
            "resolution_vertical": 480,
            "fov": 87,
            "scaling_factor": 8,
            "noiseStd": 0.0,
        }
    ]
}

class rayCreator:
    def __init__(self, sensorConfig: SensorType) -> None:
        self.sensor: SensorType = sensorConfig # Sensor configuration
        self.mesh: o3d.geometry.TriangleMesh   # O3d triangleMesh containing ray directions (no typehint because of https://github.com/isl-org/Open3D/issues/3052 )
        self.loadConfig(sensorConfig)
        
    def loadConfig(self, sensorConfig): raise NotImplementedError
    def formatROSMsg(self, resultData): raise NotImplementedError
    def postProcess(self, resultData): raise NotImplementedError
           
    def computeRays(self, robotPose: ArrayListType) -> np.ndarray:
        """Calculates the scan's rays based on stored sensor configuration and current robot pose
        """
        # Copy stored mesh (in robot frame) and rotate it to odom frame
        tmpMesh = copy.deepcopy(self.mesh)
        tmpMesh.rotate(self.mesh.get_rotation_matrix_from_xyz([0, 0, robotPose[2]]))
        castDirections = np.asarray(tmpMesh.vertices)
        # Translate to odom frame
        robotPosition = np.array([robotPose[0], robotPose[1], 0], dtype=float)
        # Since the sensor position is relative to the robot base, compensate for sensor translation
        sensorPosition = robotPosition + np.array([
            np.cos(robotPose[2])*self.sensor["pose"][0] + np.sin(robotPose[2])*self.sensor["pose"][1], #TODO: test y translation, generalise to 3d transform
            np.sin(robotPose[2])*self.sensor["pose"][0] + np.cos(robotPose[2])*self.sensor["pose"][1],
            self.sensor["pose"][2]
        ], dtype=float)
        # Return array consisting of ray positions (the same for all rays) and directions
        return np.hstack((np.full(castDirections.shape, sensorPosition), castDirections))
        
    @staticmethod
    def createAngles(minAngle: float, maxAngle: float, numScans: int) -> np.ndarray:
        """Creates an array of length numScans from minAngle to maxAngle.
        """
        # Handle single angle (i.e. 2D scanner)
        if numScans <= 1: return np.array([(maxAngle - minAngle)/2])
        # Create the array
        increment = (maxAngle - minAngle)/numScans
        return np.array([ angle for angle in np.arange(minAngle, maxAngle, increment) ])

class rayCreatorLaserScan(rayCreator):
    def __init__(self, sensorConfig: SensorTypeLaserScan) -> None:
        self.sensor: SensorTypeLaserScan
        super().__init__(sensorConfig)
    
    def loadConfig(self, sensorConfig: SensorTypeLaserScan) -> None:
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
    
    def postProcess(self, resultData) -> LaserScan:
        """Applies 
        """
        scan = LaserScan()
        scan.header.frame_id = self.sensor["frame"]
        scan.angle_min = self.sensor["angle_min_horizontal"]
        scan.angle_max = self.sensor["angle_max_horizontal"]
        scan.angle_increment = (self.sensor["angle_max_horizontal"] - self.sensor["angle_min_horizontal"])/self.sensor["num_scans_horizontal"]
        scan.time_increment = 0.0
        scan.scan_time = 0.001
        scan.range_min = 0.01
        scan.range_max = 7.0
        scan.ranges = (resultData["t_hit"]).tolist()
        return scan
    
    def formatROSMsg(self, resultData) -> LaserScan: return self.postProcess(resultData) #TODO: decouple post processing from ROS message formatting

class rayCreatorPointCloud2(rayCreatorLaserScan):
    def __init__(self, sensorConfig: SensorTypeLaserScan) -> None:
        self.sensor: SensorTypeLaserScan
        super().__init__(sensorConfig)
        
    def postProcess(self, resultData) -> PointCloud2:
        # Convert ray length to point in 3d space
        hit = np.isfinite(resultData['t_hit'])
        points = resultData["rays"][hit][:,:3] + resultData["rays"][hit][:,3:]*resultData['t_hit'][hit].reshape((-1,1))
        # Track points in o3d point cloud and convert to ros point cloud
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        return conversion.o3dCloud2pcl(pcd, frame_id="odom") #TODO: publish in frame sensor["frame"])
    
    def formatROSMsg(self, resultData) -> PointCloud2: return self.postProcess(resultData) #TODO: decouple post processing from ROS message formatting

class rayCreatorDepthCamera(rayCreator):
    def __init__(self, sensorConfig: SensorTypeDepthCamera) -> None:
        self.sensor: SensorTypeDepthCamera
        super().__init__(sensorConfig)
        
    def loadConfig(self, sensorConfig: SensorTypeDepthCamera) -> None:
        """Stores an (updated) sensor config and caches config-dependent raycast directions
        """
        self.sensor = sensorConfig
        ## Create correctly oriented ray directions and store them so we don't have to compute them each update
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=self.sensor["fov"], eye=[0, 0, 0], center=[1, 0, 0], up=[0, 0, 1],
            width_px=int(self.sensor["resolution_horizontal"]/self.sensor["scaling_factor"]),
            height_px=int(self.sensor["resolution_vertical"]/self.sensor["scaling_factor"]),
        )
        # Only the direction portion of the created array is used, since the position is transformed with the robot pose
        castDirections = rays.numpy().reshape([-1, 6])[:,3:]
        # Create o3d mesh object containing rays and rotate according to sensor orientation
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(np.array(castDirections).reshape([-1, 3]))
        self.mesh.rotate(self.mesh.get_rotation_matrix_from_xyz(self.sensor["pose"][3:6]))
    
    def postProcess(self, resultData) -> npt.NDArray:
        """Formats ROS message from raycast result
        """
        # Reshape from list to image and invert both axes due to camera model
        img = np.flip(np.flip(
            resultData["t_hit"].reshape(
                (int(self.sensor["resolution_vertical"]/self.sensor["scaling_factor"]),
                 int(self.sensor["resolution_horizontal"]/self.sensor["scaling_factor"]), -1)
            ),
        axis=1), axis=0)
        # Correct distance of not detected objects to mimic realsense behaviour
        img[img == np.inf] = 0
        # Return upsampled image (upsampling is used to save on performance TODO: noise
        upsampled = cv2.resize(img, np.dot(np.flip(img.shape[:-1]), self.sensor["scaling_factor"]), interpolation= cv2.INTER_CUBIC)
        upsampled = cv2.bilateralFilter(upsampled.astype(np.float32),4,15,15)
        return upsampled
    
    def formatROSMsg(self, resultData) -> Image:
        """Formats ROS message from raycast result
        """
        upsampled = self.postProcess(resultData)
        cv_bridge = CvBridge()
        return cv_bridge.cv2_to_imgmsg(upsampled.astype(np.float32), encoding="passthrough")
    
def rayCreatorFactory(sensorConfig) -> rayCreator:
    switchTypes = {
        "LaserScan": rayCreatorLaserScan,
        "PointCloud2": rayCreatorPointCloud2,
        "Image": rayCreatorDepthCamera
    } #This is technically not type safe, since sensorConfig["dType"] could be wrong. Therefore no type check of argument 1
    return switchTypes[sensorConfig["dType"]](sensorConfig)

def formatROSTransform(frame: str, childFrame: str, pose: ArrayListType, stamp: Optional[Time]=None) -> TransformStamped:
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
    if stamp is not None: t.header.stamp = stamp
    return t

def movementUpdate(pose: ArrayListType, cmd_vel: Twist, dt: float) -> Tuple[ArrayListType, float, float]:
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
        self.robot: RobotType
        self.robotPose: ArrayListType = robotConfig["initial_pose"]
        self.sensors: List[rayCreator]
        self.poses: List[TransformStamped]
        self.mesh: o3d.geometry.TriangleMesh
        self.loadConfig(robotConfig)
                       
        # Create raycasting scene and store mesh for visualisation
        mapVertices, mapTriangles = mapLoader.reconstruct2DMap(self.robot["map_file"])
        self.loadGeometry(mapVertices, mapTriangles)
        # Placeholder for raycasting results
        self.cachedRayList: ArrayListType
        self.cachedtHitSections: ArrayListType
        
    def loadConfig(self, robotConfig: RobotType) -> None:
        """Stores an (updated) robot config and caches config-dependent members
        """
        #TODO: instead of re-instantiating every sensor, check what has changed in the config and use sensor.loadConfig
        # Determine difference between configs: https://stackoverflow.com/questions/32815640/how-to-get-the-difference-between-two-dictionaries-in-python
        self.robot = robotConfig
        ## Instantiate rayCreators
        self.sensors = [ rayCreatorFactory(sensor) for sensor in self.robot["sensors"] ]
        # Store static transforms, since the only changing transform is odom -> base_link
        self.poses = [ formatROSTransform("base_link", sensor["frame"], sensor["pose"]) for sensor in self.robot["sensors"] ]
        
    def loadGeometry(self, mapVertices: ArrayListType, mapTriangles: ArrayListType) -> None:
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
        return self.mesh.fill_holes()
               
    def update(self, cmd_vel: Twist) -> Tuple[List[TransformStamped], ArrayListType]:
        """Performs a discrete simulation update based on the provided command velocity (will hopefully change a lot to be cleaner)
        """
        # Apply motion and sensor updates
        newPose, travelDistance, rotationDistance = movementUpdate(self.robotPose, cmd_vel, robot["dt"])
        
        # Only perform scan update if robot has moved
        if travelDistance > 0 or rotationDistance > 0 or not hasattr(self, "cachedRayList"):          
            # Update Sensor poses
            newRayList = [ sensor.computeRays(newPose) for sensor in self.sensors ]
            sections = np.cumsum([ len(entry) for entry in newRayList ])
            rayListFlat = np.concatenate(newRayList).astype(np.float32)
            # Call raycast
            rayTensor = o3d.core.Tensor(rayListFlat, dtype = o3d.core.Dtype.Float32)
            raycastResult = self.raycastingScene.cast_rays(rayTensor)
            newtHitSections = np.split(raycastResult["t_hit"].numpy(), sections)
        
            # Only apply update permanently if robot is not colliding
            #TODO find a better, actual solution in movement update
            if np.amin(newtHitSections[robot["collision_sensor"]]) > self.robot["robot_base_radius"]:
                self.robotPose = newPose
                self.cachedRayList = newRayList
                self.cachedtHitSections = newtHitSections
        
        # Create scanner messages and add gaussian noise to simulate sensor error
        pubMessages = [
            sensor.formatROSMsg({
                "t_hit": self.cachedtHitSections[i] + np.random.normal(
                    0.0, sensor.sensor["noiseStd"], size=len(self.cachedtHitSections[i])
                ),
                "rays": self.cachedRayList[i]
            })
            for i, sensor in enumerate(self.sensors)
        ]
        
        # Return all resulting transformations and messages
        retTransforms = self.poses + [formatROSTransform("odom", "base_link", [self.robotPose[0], self.robotPose[1], 0, 0, 0, self.robotPose[2]])]   
        return retTransforms, pubMessages
        
class simNode(rclpy.node.Node):
    def __init__(self, robotConfig: RobotType) -> None:
        super().__init__("{}_sim_scene".format(robot["name"]))
        self.simScene = simulationScene(robotConfig)
        
        # Sensor and transform publishers
        self.tfPub = TransformBroadcaster(self)
        self.sensorPubs = [
            self.create_publisher(eval(sensor["dType"]), sensor["topic"], 1)
            for sensor in robotConfig["sensors"]
        ]
        
        # Mesh visualisation publisher
        self.meshPub = self.create_publisher(Marker, "{}/mesh".format(robotConfig["name"]), 1)
        newMesh = self.simScene.getGeometry()
        # TODO: change mesh2ROSMeshMarker to take o3d mesh directly
        self.marker = conversion.mesh2ROSMeshMarker(newMesh.vertex.positions.numpy(), newMesh.triangle.indices.numpy(), robot["name"])
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.meshPub.publish(self.marker)
        
        # Command subscriber
        self.latestCommand = None
        self.commandSub = self.create_subscription(Twist, "{}/cmd_vel".format(robot["name"]), self.commandCallback, 1)
        
        # Update timer
        self.timer = self.create_timer(robot["dt"], self.timerCallback)
        
    def commandCallback(self, data: Twist) -> None:
        """Stores latest cmd_vel
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
    #TODO: Check if map_file is yaml or point_cloud and load map/pointcloud accordingly
    #TODO: Pull config from ROS parameter server: https://docs.ros.org/en/rolling/Tutorials/Beginner-Client-Libraries/Using-Parameters-In-A-Class-Python.html and http://design.ros2.org/articles/ros_parameters.html
    #TODO: json 2 rosparam
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
