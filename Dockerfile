FROM juliussin/algo_ultralytics:0.1.0

# Set working directory
WORKDIR /home/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tmux \
    default-libmysqlclient-dev \
    libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir mysqlclient \
    python-dotenv zmq cython_bbox ffmpeg-python tritonclient[all]

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Copy all local files into the container
COPY . .

# Ensure resource directories exist
RUN mkdir -p /home/resources /home/videos

# Start tmux sessions for ZMQ server and main logic
# CMD ["sh", "-c", "tmux new-session -d -s yolo_zmq_server './yolo_ultra/run_zmq.sh'; tmux new-session -d -s main_run './run.sh'; tail -f /dev/null"]
CMD ["bash", "./run.sh"]
