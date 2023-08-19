# tinysim-2d

Lightweight time-discrete mobile robot simulator for ROS 2 (ROS 1 in the works). The goal of this simulator is to be able to simulate the essentials of a mobile robot system with resonable accuracy and great speed. The simulator is written in Python and utilises [Open3D](http://www.open3d.org/) for sensor data simulation. The simulation scene (in `./tinysim-2d/simulatorNode.py`) can either be started as a ROS 2 node (`python ./tinysim-2d/simulatorNode.py`) or imported for use in e.g. a reinforcment learning environment.

Currently supported sensors are 3D and 2D Lidars and Depth Cameras. See the following demo of simulation scene reconstructed from a 2D map and visualised using [Foxglove](https://foxglove.dev/). The simulated sensors are the Turtlebot 3's 2D Lidar, a Velodyne Puck and the depth channel of a Realsense D435i.

<img src="./doc/simulatorDemo.gif" alt="Simulator Demo" style="max-width:=1100px" />

## Current Roadmap

There is a lot to do! These are the things I am currently working on.

- Treat robot pose and accompanying transformations as 3D instead of 2D
- Utilise [Project Chrono](https://projectchrono.org/) for physics

**These features are nice to have but take a lot of time/effort**

- Reconstruction of simulation scenes from 3D pointclouds/rosbags
- IMU simulation
- ROS parameter-driven configuration 
- Simulated colour image
- Realistic noise/error for all sensor types (maybe using GAN for depth camera?)

**Administrative TODO's**

- Releases (as Python package?)
- Github workflow for mypy checking and unit tests
- Roslaunch