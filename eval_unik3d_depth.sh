#!/bin/bash

# Create a log directory for outputs
LOG_DIR="/media/hdd2/users/haowei/Evaluation_logs"
mkdir -p $LOG_DIR

# Log file path with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/depth_evaluation_${TIMESTAMP}.log"

# Add header to log file
echo "Starting depth evaluation job at $(date)" > $MAIN_LOG
echo "=========================================" >> $MAIN_LOG

# Base directories
REPLICA_DATA_DIR="/media/hdd2/users/haowei/Replica"
REPLICA_UNIK3D_DIR="/media/hdd2/users/haowei/Replica_unik3d"

# Scene directories
SCENES=("office0" "office1" "office2" "office3" "office4" "room0" "room1" "room2")

# Subsample levels (original + subsampled versions)
SUBSAMPLE_LEVELS=("" "_2subsample" "_4subsample" "_8subsample")

# Build array of commands
COMMANDS=()

# Generate commands for all combinations
for SCENE in "${SCENES[@]}"; do
    for LEVEL in "${SUBSAMPLE_LEVELS[@]}"; do
        # Define suffix for directories
        suffix=$LEVEL
        
        # Input paths from original Replica data
        RGB_FOLDER="${REPLICA_DATA_DIR}/${SCENE}/frames${suffix}"
        SEG_MASK_FOLDER="${REPLICA_DATA_DIR}/${SCENE}/instance_colors${suffix}"
        INST_ID_FOLDER="${REPLICA_DATA_DIR}/${SCENE}/vis_instance_ids${suffix}"
        GT_DEPTH_FOLDER="${REPLICA_DATA_DIR}/${SCENE}/depths"
        
        # Prediction paths from Replica_unik3d (matching the exact directory structure shown)
        PRED_DEPTH_RGB_FOLDER="${REPLICA_UNIK3D_DIR}/${SCENE}/frames${suffix}"
        PRED_DEPTH_SEG_FOLDER="${REPLICA_UNIK3D_DIR}/${SCENE}/instance_colors${suffix}"
        PRED_DEPTH_INST_FOLDER="${REPLICA_UNIK3D_DIR}/${SCENE}/instance_ids${suffix}"
        
        # Output path for comparison results
        OUTPUT_FOLDER="${REPLICA_UNIK3D_DIR}/${SCENE}/comparison${suffix}"
        mkdir -p $OUTPUT_FOLDER
        
        # Skip if any input directory doesn't exist
        if [ ! -d "$RGB_FOLDER" ] || [ ! -d "$SEG_MASK_FOLDER" ] || [ ! -d "$INST_ID_FOLDER" ] || [ ! -d "$GT_DEPTH_FOLDER" ]; then
            echo "Skipping due to missing input directory for ${SCENE}${suffix}" >> $MAIN_LOG
            continue
        fi
        
        # Skip if any prediction directory doesn't exist
        if [ ! -d "$PRED_DEPTH_RGB_FOLDER" ] || [ ! -d "$PRED_DEPTH_SEG_FOLDER" ] || [ ! -d "$PRED_DEPTH_INST_FOLDER" ]; then
            echo "Skipping due to missing prediction directory for ${SCENE}${suffix}" >> $MAIN_LOG
            continue
        fi
        
        # Log file for this specific task
        TASK_LOG="${LOG_DIR}/${SCENE}_evaluation${suffix}_${TIMESTAMP}.log"
        
        # Create command with output redirection
        CMD="python eval_depth.py --mode folder \
            --rgb_folder ${RGB_FOLDER} \
            --seg_mask_folder ${SEG_MASK_FOLDER} \
            --inst_id_folder ${INST_ID_FOLDER} \
            --pred_depth_rgb_folder ${PRED_DEPTH_RGB_FOLDER} \
            --pred_depth_seg_folder ${PRED_DEPTH_SEG_FOLDER} \
            --pred_depth_inst_folder ${PRED_DEPTH_INST_FOLDER} \
            --gt_depth_folder ${GT_DEPTH_FOLDER} \
            --output_folder ${OUTPUT_FOLDER} > ${TASK_LOG} 2>&1"
        
        # Add to commands array
        COMMANDS+=("$CMD")
        
        # Log the command
        echo "Added task: $CMD" >> $MAIN_LOG
    done
done

# Count total number of tasks
NUM_TASKS=${#COMMANDS[@]}
echo "Total number of tasks: $NUM_TASKS" >> $MAIN_LOG

# Function to run tasks in background with limited concurrency
run_with_limit() {
    # Maximum number of concurrent tasks
    local MAX_JOBS=8
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