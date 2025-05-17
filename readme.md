# Project Instructions

---

## Project Structure
```
project_root/
├── algo_helper/            # Contains all algorithm-related scripts
│   ├── use/                # Additional libraries for algorithms
│   ├── <used_va_scripts>.py  # Only necessary VA scripts
│   ├── _alert_va.py        # Alert library (handles MySQL updates & image saving in threading)
│
├── client_scripts/         # Scripts for inferencing models
│
├── docker/                 # Model server scripts
│   ├── run.sh              # Run script (change Docker container name first!)
│
├── tracking/               # Tracker implementation (ByteTrack)
│
├── triton_client_script/   # YOLO object detection script
│
├── utils/                  # Additional libraries for run_step(1,2,3).py
│
├── _tmp/                   # Backup folder
│
├── run_step1_restart.py    # Fixed script, do not modify
├── run_step2_sql.py        # Fixed script, do not modify
├── run_step3.py            # Main execution script
│
├── requirements.txt        # Required dependencies
├── run.sh                  # Script to run everything
├── .env                    # Environment configurations
└── README.md               # Documentation
```

---

## Usage
- `run_step1_restart.py` and `run_step2_sql.py` **must not be modified** as they are fixed.
- Modify and update `run_step3.py` as necessary for execution.
- Keep all additional utilities in the `utils/` folder.
- Ensure that all **algorithm-related** scripts are inside `algo_helper/`.
- Use `triton_client_script/` for YOLO object detection.
- The `tracking/` folder is strictly for tracking implementations (ByteTrack).
- Change the **Docker container name** before running `docker/run.sh`.

### Commands:
```sh
docker/run.sh  # Change Docker container name first!
pip3 install -r requirements.txt
./run.sh
```

---

**Warnings & Important Notes:**
- `run_step1_restart.py` and `run_step2_sql.py` **must not be modified**.
- **Set environment variables in `.env`** (not detailed here).
- **Only include necessary VA scripts** in `algo_helper/`.
- **Change the Docker container name** before running `docker/run.sh`.
- `_alert_va.py` is responsible for MySQL updates and image saving in a separate thread—use it accordingly.

---

## Folder Details
### `utils/`
- Contains additional libraries needed for `run_step1_restart.py`, `run_step2_sql.py`, and `run_step3.py`.
- **Not for algorithm-related code**.

### `algo_helper/`
- Stores all algorithm-related scripts.
- The `use/` subfolder contains additional libraries required for the algorithm.
- **Only include necessary VA scripts** in this folder to avoid clutter.
- `_alert_va.py` handles MySQL updates and image saving using threading.

### `client_scripts/`
- Contains scripts specifically for inferencing models.

### `triton_client_script/`
- Contains YOLO object detection scripts.

### `tracking/`
- Contains tracker implementation (ByteTrack).

### `docker/`
- Holds the model server scripts for deployment.
- **Change the Docker container name** before executing `run.sh`.

### `_tmp/`
- Backup storage location.
- Use this folder to store backups when necessary.

---

## Backup Policy
- Store any necessary backup files in the `_tmp/` folder.
- Ensure that important scripts are version-controlled properly.
- Avoid cluttering algorithm-related folders with unused scripts.


