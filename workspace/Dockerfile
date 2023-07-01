# Pull ros-ml-container from container registry
FROM ghcr.io/simonschwaiger/ros-ml-container:ros2_humble

# Update cache and install turtlebot requirements
RUN apt update && apt install -y --no-install-recommends \
    ros-humble-rviz2
    # ros-noetic-turtlebot3-simulations
    #ros-noetic-amcl ros-noetic-map-server ros-noetic-move-base \
    #ros-noetic-turtlebot3-msgs ros-noetic-rosserial-python \
    #ros-noetic-hls-lfcd-lds-driver 

# Install additional requirements from local file
RUN rm requirements.txt
ADD ./requirements.txt .
RUN /bin/bash -c "source ~/myenv/bin/activate \
    && pip3 install -r requirements.txt"

# Copy ROS packages for compilation in container
COPY ./src /opt/$ROS2_WS/src

# Install ros dependencies
RUN apt-get update && rosdep update && rosdep install --from-paths /opt/$ROS2_WS/src -i -y --rosdistro $ROS_DISTRO

# Compile workspace
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash \
    && cd /opt/$ROS2_WS \
    && colcon build --symlink-install"

# Remove src folder used for compilation, since the real src folder will be mounted at runtime
RUN rm -rf /opt/$ROS2_WS/src

# Cleanup
RUN rm -rf /var/lib/apt/lists/*