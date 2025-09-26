#!/usr/bin/env bash
set -e

export DISPLAY=:1

echo "🔧 Starting virtual X server..."
Xvfb :1 -screen 0 1280x800x24 &
sleep 2

echo "🖥️ Launching XFCE desktop..."
startxfce4 &
sleep 2

echo "🔒 Starting x11vnc server..."
x11vnc -display :1 -rfbport 5901 -forever -shared -passwd 1234 > /tmp/x11vnc.log 2>&1 &
sleep 2

# Confirm VNC is listening
if netstat -tuln | grep -q ":5901"; then
  echo "✅ VNC server is listening on port 5901"
else
  echo "❌ VNC server failed to start"
  echo "🔍 x11vnc log:"
  cat /tmp/x11vnc.log
  exit 1
fi

echo "🌐 Starting noVNC websockify..."
websockify --web=/usr/share/novnc/ 8080 localhost:5901 > /tmp/websockify.log 2>&1 &
sleep 2

# Confirm websockify is listening
if netstat -tuln | grep -q ":8080"; then
  echo "✅ noVNC is listening on port 8080"
else
  echo "❌ noVNC failed to start"
  echo "🔍 websockify log:"
  cat /tmp/websockify.log
  exit 1
fi

echo "🎉 VNC + noVNC stack is up"
echo "👉 Open http://localhost:8080/vnc.html (password: 1234)"

# Keep container alive
tail -f /dev/null