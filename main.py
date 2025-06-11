### SECTION 1: SETUP AND IMPORTS

# Import required packages - you need to install these first with pip
import sys, os, time, cv2, numpy as np, torch, tempfile, asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw
from fpdf import FPDF
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.models.video as models
import torchvision.transforms as transforms
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog
import torch

from missing_person_detection import run_missing_person_detection
from violence_detection import run_violence_detection
from utils import select_files, load_video_files, load_reference_files

# Import Spark processing functions
from spark_processing import run_distributed_pipeline,initialize_spark,load_reference_images

### SECTION 4: INTEGRATION AND MAIN EXECUTION

def run_full_pipeline():
    """Run both missing person detection and violence detection pipelines"""
    print("======================================================")
    print("MISSING PERSON AND VIOLENCE DETECTION SYSTEM")
    print("======================================================")
    print("This system combines two detection capabilities:")
    print("1. Missing Person Detection: Identifies a specific person in videos")
    print("2. Violence Detection: Identifies potential violent content in videos")
    print("======================================================")

    # First, run missing person detection
    print("\n[STEP 1] MISSING PERSON DETECTION")
    print("------------------------------")
    missing_person_detections = run_missing_person_detection()

    # Then, proceed with violence detection
    if missing_person_detections:
        print("\n[STEP 2] VIOLENCE DETECTION")
        print("------------------------------")
        # Extract unique video files from detections
        detected_videos = list(set([det['video_filename'] for det in missing_person_detections]))
        print(f"Analyzing {len(detected_videos)} videos with detected missing persons for violence...")
        run_violence_detection(detected_videos)
    else:
        print("\nNo missing person detections to analyze for violence.")

    print("\n======================================================")
    print("ANALYSIS COMPLETE")
    print("======================================================")
    print("If the missing person was found, please contact authorities immediately.")
    print("If violence was detected, please review the reports carefully and take appropriate action.")

### SECTION 5: INDIVIDUAL MODULE EXECUTION

def run_only_missing_person_detection():
    """Run only the missing person detection module"""
    print("Running Missing Person Detection Only")
    run_missing_person_detection()

def run_only_violence_detection():
    """Run only the violence detection module"""
    print("Running Violence Detection Only")
    # For local environment, use file dialog instead of files.upload()
    print("Select video files to analyze for violence:")
    video_files = select_files("Select Video Files for Violence Detection", 
                             [("Video files", "*.mp4 *.avi *.mov")])
    
    if video_files:
        run_violence_detection(video_files)
    else:
        print("No videos selected. Exiting.")


# Main execution section
if __name__ == '__main__':
    print("======================================================")
    print("CCTV ANALYSIS SYSTEM FOR LARGE DATASETS")
    print("======================================================")
    print("Select an option:")
    print("1. Run full pipeline (Missing Person + Violence Detection)")
    print("2. Run only Missing Person Detection")
    print("3. Run only Violence Detection")
    print("4. Run Spark Distributed Pipeline")
    
    try:
        choice = int(input("Enter your choice (1-4): "))
        if choice == 1:
            run_full_pipeline()
        elif choice == 2:
            run_only_missing_person_detection()
        elif choice == 3:
            run_only_violence_detection()
        elif choice == 4:
            spark=initialize_spark()
            ref_files=load_reference_files()
            ref_img=load_reference_images(ref_files)
            video_files=load_video_files()
            run_distributed_pipeline(spark,video_files,ref_img)
        else:
            print("Invalid choice. Exiting.")
    except ValueError:
        print("Please enter a number between 1 and 4. Exiting.")