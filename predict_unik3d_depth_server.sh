#!/bin/bash

# Create a log directory for outputs
LOG_DIR="/media/hdd2/users/haowei/Replica_logs"
mkdir -p $LOG_DIR

# Log file path with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/depth_prediction_${TIMESTAMP}.log"

# Add header to log file
echo "Starting depth prediction job at $(date)" > $MAIN_LOG
echo "=========================================" >> $MAIN_LOG

# Add warning filter to suppress KNN messages
export PYTHONWARNINGS="ignore::UserWarning:.*KNN.*"

# Base directories
INPUT_BASE="/media/hdd2/users/haowei/Replica"
OUTPUT_BASE="/media/hdd2/users/haowei/Replica_output"
CAMERA_PATH="/media/hdd2/users/haowei/Replica/cam_params.json"
CONFIG_FILE="configs/eval/vitl.json"

# Create output base directory if it doesn't exist
mkdir -p $OUTPUT_BASE

# Scene directories
SCENES=("office0" "office1" "office2" "office3" "office4" "room0" "room1" "room2")

# Image types
TYPES=("frames" "instance_colors" "instance_ids")

# Subsample levels (original + subsampled versions)
SUBSAMPLE_LEVELS=("" "_2subsample" "_4subsample" "_8subsample")

# Build array of commands
COMMANDS=()

# Generate commands for all combinations
for SCENE in "${SCENES[@]}"; do
    for TYPE in "${TYPES[@]}"; do
        for LEVEL in "${SUBSAMPLE_LEVELS[@]}"; do
            # Input folder path
            INPUT_FOLDER="${INPUT_BASE}/${SCENE}/${TYPE}${LEVEL}"
            
            # Skip if input directory doesn't exist
            if [ ! -d "$INPUT_FOLDER" ]; then
                echo "Skipping non-existent directory: $INPUT_FOLDER" >> $MAIN_LOG
                continue
            fi
            
            # Create output folder
            OUTPUT_FOLDER="${OUTPUT_BASE}/${SCENE}/${TYPE}${LEVEL}"
            mkdir -p $OUTPUT_FOLDER
            
            # Log file for this specific task
            TASK_LOG="${LOG_DIR}/${SCENE}_${TYPE}${LEVEL}_${TIMESTAMP}.log"
            
            # Create command with output redirection
            CMD="python predict_unik3d_depth.py --mode folder --rgb_folder ${INPUT_FOLDER} --output_folder ${OUTPUT_FOLDER} --camera_path ${CAMERA_PATH} --config_file ${CONFIG_FILE} > ${TASK_LOG} 2>&1"
            
            # Add to commands array
            COMMANDS+=("$CMD")
            
            # Log the command
            echo "Added task: $CMD" >> $MAIN_LOG
        done
    done
done

# Count total number of tasks
NUM_TASKS=${#COMMANDS[@]}
echo "Total number of tasks: $NUM_TASKS" >> $MAIN_LOG

# Function to run tasks in background with limited concurrency
run_with_limit() {
    # Maximum number of concurrent tasks
    local MAX_JOBS=4
    # Current running jobs count
    local RUNNING=0
    # Array of background process IDs
    local PIDS=()
    
    echo "Starting execution at $(date) with max $MAX_JOBS concurrent jobs" >> $MAIN_LOG
    
    # Process each command
    for CMD in "${COMMANDS[@]}"; do
        # Wait if we have reached max concurrent jobs
        while [ $RUNNING -ge $MAX_JOBS ]; do
            # Check for completed jobs
            for i in "${!PIDS[@]}"; do
                if ! kill -0 ${PIDS[$i]} 2>/dev/null; then
                    # Process has completed
                    unset PIDS[$i]
                    ((RUNNING--))
                    echo "A task completed. Running: $RUNNING/$MAX_JOBS" >> $MAIN_LOG
                fi
            done
            # Short sleep to prevent CPU hogging
            sleep 2
        done
        
        # Start a new task
        eval "$CMD" &
        PIDS+=($!)
        ((RUNNING++))
        echo "Started new task (PID: ${PIDS[-1]}). Running: $RUNNING/$MAX_JOBS" >> $MAIN_LOG
        # Short sleep to prevent starting too many jobs at once
        sleep 1
    done
    
    # Wait for all remaining jobs to complete
    echo "All tasks started. Waiting for remaining $RUNNING tasks to complete..." >> $MAIN_LOG
    for PID in "${PIDS[@]}"; do
        if [ -n "$PID" ]; then
            wait $PID
            echo "Task with PID $PID completed" >> $MAIN_LOG
        fi
    done
    
    echo "All tasks completed at $(date)" >> $MAIN_LOG
}

# Run the tasks in background
(run_with_limit >> $MAIN_LOG 2>&1) &

# Get the process ID
PROCESS_ID=$!
echo "Started background process with PID: $PROCESS_ID" >> $MAIN_LOG
echo "Started background process with PID: $PROCESS_ID"
echo "Log file: $MAIN_LOG"
echo "You can now disconnect from the server. The process will continue running."
echo "To check status later, use: tail -f $MAIN_LOG"