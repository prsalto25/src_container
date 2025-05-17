script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"

portIn=4000     # client (send img -> get pred)
portOut=4001    # server (get img -> send pred)

python3 broker.py $portIn, $portOut & 
python3 yolo_ultralytics_zmq.py