# File: Dockerfile
FROM --platform=linux/arm64 osrf/ros:humble-desktop-full

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TURTLEBOT3_MODEL=burger

# -------------------------
# System & ROS2 dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    python3-pip python3-colcon-common-extensions \
    tree \
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    ros-humble-rqt-graph \
    ros-humble-rviz2 \
    ros-humble-gazebo-plugins \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-turtlebot3-gazebo \
    ros-humble-turtlebot3-teleop \
    build-essential git curl wget vim \
    net-tools htop tmux \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Python packages for RL
# -------------------------
RUN pip3 install --no-cache-dir \
    numpy pandas opencv-python matplotlib \
    torch torchvision \
    gym==0.26.2 gymnasium \
    stable-baselines3[extra] \
    optuna tensorboard

# -------------------------
# VNC + noVNC setup
# -------------------------
RUN apt-get update && apt-get install -y \
    xfce4 xfce4-goodies \
    novnc websockify \
    x11vnc xvfb supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create VNC startup script
RUN mkdir -p /root/.vnc && \
    x11vnc -storepasswd 1234 /root/.vnc/passwd

# Create noVNC startup script
RUN mkdir -p /opt/scripts
COPY .devcontainer/start-vnc.sh /opt/scripts/start-vnc.sh
RUN chmod +x /opt/scripts/start-vnc.sh

# Expose VNC and noVNC ports
EXPOSE 5901 8080

# Default workdir
WORKDIR /workspace

# Set default shell and ROS2 source
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc && \
    echo "source /workspace/install/setup.bash" >> ~/.bashrc
