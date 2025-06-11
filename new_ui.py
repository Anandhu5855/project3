import platform
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import torch
from PIL import Image, ImageTk
import numpy as np
import time

# Import functions from the new modules
from missing_person_detection import setup_missing_person_detection, process_video
from violence_detection import load_violence_detection_model, detect_violence_in_video
from report_generation import export_to_pdf, export_violence_report,generate_combined_report
from spark_processing import (
    initialize_spark,
    run_distributed_pipeline
)
from stats import stats_monitor,calculate_performance_stats
from config import config

# Import for Spark distributed processing
try:
    import pyspark
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

class DarkTheme:
    """Theme colors for the dark UI"""
    BG_DARK = "#121212"
    BG_MEDIUM = "#1E1E1E"
    BG_LIGHT = "#2D2D2D"
    TEXT = "#FFFFFF"
    ACCENT = "#7E57C2"  # Purple accent
    SUCCESS = "#4CAF50"  # Green
    ERROR = "#F44336"    # Red
    WARNING = "#FF9800"  # Orange
    INFO = "#2196F3"     # Blue

class MissingPersonDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Missing Person & Violence Detection")
        self.root.geometry("1100x800")
        self.root.configure(bg=DarkTheme.BG_DARK)   
        # Apply dark theme to ttk widgets
        self.apply_dark_style()
        
        # Variables
        self.ref_files = []
        self.video_files = []
        self.metrics_vars = {
        'accuracy': tk.StringVar(value="Accuracy: -"),
        'precision': tk.StringVar(value="Precision: -"), 
        'recall': tk.StringVar(value="Recall: -"),
        'fps': tk.StringVar(value="FPS: -"),
        'detections': tk.StringVar(value="Detections: TP=0 FP=0"),
        'total_time': tk.StringVar(value="Total Time: -"),  # Added
        'total_frames': tk.StringVar(value="Frames: -")     # Added
        }
        self.detection_threshold = tk.DoubleVar(value=0.65)
        self.frame_interval = tk.IntVar(value=60)
        self.use_gpu = tk.BooleanVar(value=torch.cuda.is_available())
        self.running = False
        self.status_text = tk.StringVar(value="Ready")
        self.start_time = None  # To track the start time of detection
        self.num_detections = 0  # To count the number of detections
        self.recommendation_var = tk.StringVar(value="Recommendations: -")
        # Spark configuration variables
        self.spark_master = tk.StringVar(value=config.SPARK_CONF.get("master", "local[*]"))
        self.spark_executor_memory = tk.StringVar(value=config.SPARK_CONF.get("executor.memory", "4g"))
        self.spark_executor_cores = tk.IntVar(value=int(config.SPARK_CONF.get("executor.cores", 2)))
        self.spark_executor_instances = tk.IntVar(value=int(config.SPARK_CONF.get("spark.executor.instances", 2)))

        self.metrics_vars.update({
            'total_time': tk.StringVar(value="Total Time: -"),
            'total_frames': tk.StringVar(value="Frames: -")
        }) 
        # Create UI
        self.create_header()
        self.create_main_frame()
        self.create_status_bar()
        # self.create_performance_panel(content)
    
    def apply_dark_style(self):
        # Configure ttk styles
        style = ttk.Style()
        
        # Use 'clam' theme as base
        style.theme_use('clam')
        
        # Configure TCombobox style
        style.configure('TCombobox', 
                        fieldbackground=DarkTheme.BG_LIGHT,
                        background=DarkTheme.BG_LIGHT,
                        foreground=DarkTheme.TEXT,
                        arrowcolor=DarkTheme.ACCENT)
        
        # Configure TProgressbar style
        style.configure("TProgressbar", 
                        troughcolor=DarkTheme.BG_LIGHT,
                        background=DarkTheme.ACCENT)
        
        # Notebook style
        style.configure("TNotebook", 
                        background=DarkTheme.BG_DARK, 
                        tabmargins=[2, 5, 2, 0])
        
        style.configure("TNotebook.Tab", 
                        background=DarkTheme.BG_LIGHT,
                        foreground=DarkTheme.TEXT,
                        padding=[10, 4],
                        font=('Arial', 10))
        
        style.map("TNotebook.Tab",
                  background=[("selected", DarkTheme.ACCENT)],
                  foreground=[("selected", DarkTheme.TEXT)])
        
    def create_header(self):
        header_frame = tk.Frame(self.root, bg=DarkTheme.BG_MEDIUM, height=70)
        header_frame.pack(fill=tk.X)
        
        # Logo or icon placeholder
        logo_label = tk.Label(
            header_frame,
            text="🔍",
            font=("Arial", 24),
            fg=DarkTheme.ACCENT,
            bg=DarkTheme.BG_MEDIUM,
            padx=20
        )
        logo_label.pack(side=tk.LEFT)
        
        title_label = tk.Label(
            header_frame, 
            text="Advanced Detection System", 
            font=("Arial", 22, "bold"),
            fg=DarkTheme.TEXT,
            bg=DarkTheme.BG_MEDIUM,
            pady=15
        )
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = tk.Label(
            header_frame, 
            text="PRID", 
            font=("Arial", 12),
            fg="#AAAAAA",
            bg=DarkTheme.BG_MEDIUM,
            pady=15,
            padx=10
        )
        subtitle_label.pack(side=tk.LEFT)
        
    def create_main_frame(self):
        # Create a notebook with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main tab for detection
        main_tab = tk.Frame(self.notebook, bg=DarkTheme.BG_DARK)
        self.notebook.add(main_tab, text="Detection")
        
        # Settings tab for advanced configuration
        settings_tab = tk.Frame(self.notebook, bg=DarkTheme.BG_DARK)
        self.notebook.add(settings_tab, text="Advanced Settings")
        
        # Main tab content
        main_frame = tk.Frame(main_tab, bg=DarkTheme.BG_DARK, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (settings)
        settings_frame = self.create_labeled_frame(main_frame, "Settings", 0.4)
        settings_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Right panel (file selection and preview)
        files_frame = self.create_labeled_frame(main_frame, "Files", 0.6)
        files_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Settings components
        self.create_settings_components(settings_frame)
        
        # Files components
        self.create_files_components(files_frame)
        
        # Advanced settings tab content
        self.create_advanced_settings(settings_tab)
        
    def create_labeled_frame(self, parent, title, weight=1):
        """Create a frame with a modern label design"""
        frame = tk.Frame(parent, bg=DarkTheme.BG_DARK, padx=5, pady=5)
        
        # Add a title bar
        title_bar = tk.Frame(frame, bg=DarkTheme.ACCENT, height=30)
        title_bar.pack(fill=tk.X)
        
        title_label = tk.Label(
            title_bar,
            text=title,
            font=("Arial", 12, "bold"),
            fg=DarkTheme.TEXT,
            bg=DarkTheme.ACCENT,
            pady=5
        )
        title_label.pack(side=tk.LEFT, padx=10)
        
        # Main content area
        content_frame = tk.Frame(frame, bg=DarkTheme.BG_MEDIUM, padx=10, pady=10)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Store the content frame as an attribute of the main frame
        frame.content = content_frame
        
        return frame
    
    def create_settings_components(self, parent):
        content = parent.content
        
        # Detection type with styled frame
        detection_frame = tk.Frame(content, bg=DarkTheme.BG_MEDIUM, pady=5)
        detection_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            detection_frame, 
            text="Detection Mode:", 
            bg=DarkTheme.BG_MEDIUM,
            fg=DarkTheme.TEXT,
            font=("Arial", 10)
        ).pack(side=tk.LEFT)
        
        self.detection_mode = tk.StringVar(value="Full Pipeline")
        modes = ["Full Pipeline", "Missing Person Only", "Violence Only", "Spark Distributed"]
        
        # If Spark is not available, disable the distributed option
        if not SPARK_AVAILABLE:
            modes = [m for m in modes if m != "Spark Distributed"]
        
        mode_combo = ttk.Combobox(
            detection_frame, 
            textvariable=self.detection_mode, 
            values=modes, 
            state="readonly", 
            width=15
        )
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)
        
        # GPU checkbox with styled frame
        gpu_frame = tk.Frame(content, bg=DarkTheme.BG_MEDIUM, pady=5)
        gpu_frame.pack(fill=tk.X, pady=5)
        
        gpu_check = tk.Checkbutton(
            gpu_frame, 
            text=f"Use GPU {'(Available)' if torch.cuda.is_available() else '(Not Available)'}", 
            variable=self.use_gpu,
            bg=DarkTheme.BG_MEDIUM,
            fg=DarkTheme.TEXT,
            selectcolor=DarkTheme.BG_DARK,
            activebackground=DarkTheme.BG_MEDIUM,
            activeforeground=DarkTheme.TEXT,
            state=tk.NORMAL if torch.cuda.is_available() else tk.DISABLED
        )
        gpu_check.pack(side=tk.LEFT)
        
        # Display real-time video
        self.display_video = tk.BooleanVar(value=True)
        display_check = tk.Checkbutton(
            gpu_frame, 
            text="Display Real-Time Video", 
            variable=self.display_video,
            bg=DarkTheme.BG_MEDIUM,
            fg=DarkTheme.TEXT,
            selectcolor=DarkTheme.BG_DARK,
            activebackground=DarkTheme.BG_MEDIUM,
            activeforeground=DarkTheme.TEXT
        )
        display_check.pack(side=tk.RIGHT, padx=5)

        # Threshold slider with styled frame
        threshold_frame = tk.Frame(content, bg=DarkTheme.BG_MEDIUM, pady=5)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            threshold_frame, 
            text="Detection Threshold:", 
            bg=DarkTheme.BG_MEDIUM,
            fg=DarkTheme.TEXT,
            font=("Arial", 10)
        ).pack(anchor=tk.W)
        
        threshold_slider = tk.Scale(
            threshold_frame,
            variable=self.detection_threshold,
            from_=0.5,
            to=0.95,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            bg=DarkTheme.BG_LIGHT,
            fg=DarkTheme.TEXT,
            highlightthickness=0,
            troughcolor=DarkTheme.BG_DARK,
            activebackground=DarkTheme.ACCENT
        )
        threshold_slider.pack(fill=tk.X)

        # Frame interval slider with styled frame
        interval_frame = tk.Frame(content, bg=DarkTheme.BG_MEDIUM, pady=5)
        interval_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            interval_frame, 
            text="Frame Interval:", 
            bg=DarkTheme.BG_MEDIUM,
            fg=DarkTheme.TEXT,
            font=("Arial", 10)
        ).pack(anchor=tk.W)
        
        interval_slider = tk.Scale(
            interval_frame,
            variable=self.frame_interval,
            from_=1,
            to=300,
            resolution=5,
            orient=tk.HORIZONTAL,
            bg=DarkTheme.BG_LIGHT,
            fg=DarkTheme.TEXT,
            highlightthickness=0,
            troughcolor=DarkTheme.BG_DARK,
            activebackground=DarkTheme.ACCENT
        )
        interval_slider.pack(fill=tk.X)
        
        # Action buttons
        action_frame = tk.Frame(content, bg=DarkTheme.BG_MEDIUM, pady=10)
        action_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.run_button = tk.Button(
            action_frame,
            text="Run Detection",
            command=self.run_detection,
            bg=DarkTheme.SUCCESS,
            fg=DarkTheme.TEXT,
            font=("Arial", 12, "bold"),
            pady=10,
            relief=tk.FLAT,
            activebackground=DarkTheme.SUCCESS,
            activeforeground=DarkTheme.TEXT
        )
        self.run_button.pack(fill=tk.X, pady=5)
        
        exit_button = tk.Button(
            action_frame,
            text="Exit",
            command=self.root.destroy,
            bg=DarkTheme.ERROR,
            fg=DarkTheme.TEXT,
            font=("Arial", 12),
            pady=5,
            relief=tk.FLAT,
            activebackground=DarkTheme.ERROR,
            activeforeground=DarkTheme.TEXT
        )
        exit_button.pack(fill=tk.X, pady=5)
    
    def create_files_components(self, parent):
        content = parent.content
        
        # Reference images 
        ref_frame = tk.Frame(content, bg=DarkTheme.BG_MEDIUM)
        ref_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            ref_frame, 
            text="Reference Images:", 
            bg=DarkTheme.BG_MEDIUM,
            fg=DarkTheme.TEXT,
            font=("Arial", 10)
        ).pack(anchor=tk.W)
        
        self.ref_label = tk.Label(
            ref_frame,
            text="No images selected",
            bg=DarkTheme.BG_LIGHT, 
            fg=DarkTheme.TEXT,
            relief=tk.FLAT,
            height=2,
            anchor=tk.W,
            padx=5
        )
        self.ref_label.pack(fill=tk.X, pady=2)
        
        ref_button = tk.Button(
            ref_frame,
            text="Select Reference Images",
            command=self.select_reference_images,
            bg=DarkTheme.INFO,
            fg=DarkTheme.TEXT,
            relief=tk.FLAT,
            activebackground=DarkTheme.INFO,
            activeforeground=DarkTheme.TEXT
        )
        ref_button.pack(anchor=tk.W)
        
        # Video files
        video_frame = tk.Frame(content, bg=DarkTheme.BG_MEDIUM)
        video_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            video_frame, 
            text="Video Files:", 
            bg=DarkTheme.BG_MEDIUM,
            fg=DarkTheme.TEXT,
            font=("Arial", 10)
        ).pack(anchor=tk.W)
        
        self.video_label = tk.Label(
            video_frame,
            text="No videos selected",
            bg=DarkTheme.BG_LIGHT, 
            fg=DarkTheme.TEXT,
            relief=tk.FLAT,
            height=2,
            anchor=tk.W,
            padx=5
        )
        self.video_label.pack(fill=tk.X, pady=2)
        
        video_button = tk.Button(
            video_frame,
            text="Select Video Files",
            command=self.select_video_files,
            bg=DarkTheme.INFO,
            fg=DarkTheme.TEXT,
            relief=tk.FLAT,
            activebackground=DarkTheme.INFO,
            activeforeground=DarkTheme.TEXT
        )
        video_button.pack(anchor=tk.W)
        
        # Preview area
        preview_frame = self.create_labeled_frame(content, "Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        preview_content = preview_frame.content
        
        self.preview_label = tk.Label(
            preview_content,
            text="Select files to see preview",
            bg=DarkTheme.BG_DARK,
            fg=DarkTheme.TEXT,
            height=10
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        # Add to create_files_components method:
        self.create_performance_panel(content)
    
    def create_advanced_settings(self, parent):
        """Create advanced settings tab with Spark configuration"""
        advanced_frame = tk.Frame(parent, bg=DarkTheme.BG_DARK, padx=20, pady=20)
        advanced_frame.pack(fill=tk.BOTH, expand=True)
        
        # Spark Configuration
        spark_frame = self.create_labeled_frame(advanced_frame, "Spark Distributed Processing Configuration")
        spark_frame.pack(fill=tk.X, pady=10)
        spark_content = spark_frame.content
        
        # Spark availability message
        if not SPARK_AVAILABLE:
            warning_label = tk.Label(
                spark_content,
                text="PySpark is not installed. Distributed processing is unavailable.",
                bg=DarkTheme.WARNING,
                fg=DarkTheme.TEXT,
                padx=10,
                pady=10,
                font=("Arial", 11)
            )
            warning_label.pack(fill=tk.X, pady=10)
            
            install_info = tk.Label(
                spark_content,
                text="To enable, install PySpark using: pip install pyspark",
                bg=DarkTheme.BG_MEDIUM,
                fg=DarkTheme.TEXT,
                padx=10,
                pady=5,
                font=("Arial", 10)
            )
            install_info.pack(fill=tk.X)
            return
        
        # Spark master URL
        master_frame = tk.Frame(spark_content, bg=DarkTheme.BG_MEDIUM)
        master_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            master_frame,
            text="Spark Master URL:",
            bg=DarkTheme.BG_MEDIUM,
            fg=DarkTheme.TEXT
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Entry(
            master_frame,
            textvariable=self.spark_master,
            bg=DarkTheme.BG_LIGHT,
            fg=DarkTheme.TEXT,
            insertbackground=DarkTheme.TEXT,
            relief=tk.FLAT,
            bd=1
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Executor memory
        memory_frame = tk.Frame(spark_content, bg=DarkTheme.BG_MEDIUM)
        memory_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            memory_frame,
            text="Executor Memory:",
            bg=DarkTheme.BG_MEDIUM,
            fg=DarkTheme.TEXT
        ).pack(side=tk.LEFT, padx=5)
        
        memory_values = ["1g", "2g", "4g", "8g", "16g"]
        memory_combo = ttk.Combobox(
            memory_frame,
            textvariable=self.spark_executor_memory,
            values=memory_values,
            width=10
        )
        memory_combo.pack(side=tk.LEFT, padx=5)
        
        # Executor cores
        cores_frame = tk.Frame(spark_content, bg=DarkTheme.BG_MEDIUM)
        cores_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            cores_frame,
            text="Executor Cores:",
            bg=DarkTheme.BG_MEDIUM,
            fg=DarkTheme.TEXT
        ).pack(side=tk.LEFT, padx=5)
        
        cores_scale = tk.Scale(
            cores_frame,
            variable=self.spark_executor_cores,
            from_=1,
            to=8,
            orient=tk.HORIZONTAL,
            bg=DarkTheme.BG_LIGHT,
            fg=DarkTheme.TEXT,
            highlightthickness=0,
            troughcolor=DarkTheme.BG_DARK,
            activebackground=DarkTheme.ACCENT
        )
        cores_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Executor instances
        instances_frame = tk.Frame(spark_content, bg=DarkTheme.BG_MEDIUM)
        instances_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            instances_frame,
            text="Executor Instances:",
            bg=DarkTheme.BG_MEDIUM,
            fg=DarkTheme.TEXT
        ).pack(side=tk.LEFT, padx=5)
        
        instances_scale = tk.Scale(
            instances_frame,
            variable=self.spark_executor_instances,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            bg=DarkTheme.BG_LIGHT,
            fg=DarkTheme.TEXT,
            highlightthickness=0,
            troughcolor=DarkTheme.BG_DARK,
            activebackground=DarkTheme.ACCENT
        )
        instances_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Additional settings
        advanced_settings_frame = self.create_labeled_frame(advanced_frame, "System Settings")
        advanced_settings_frame.pack(fill=tk.X, pady=10)
        advanced_content = advanced_settings_frame.content
        
        # Cache management
        cache_frame = tk.Frame(advanced_content, bg=DarkTheme.BG_MEDIUM)
        cache_frame.pack(fill=tk.X, pady=5)
        
        self.clear_cache = tk.BooleanVar(value=True)
        cache_check = tk.Checkbutton(
            cache_frame,
            text="Clear cache after processing",
            variable=self.clear_cache,
            bg=DarkTheme.BG_MEDIUM,
            fg=DarkTheme.TEXT,
            selectcolor=DarkTheme.BG_DARK,
            activebackground=DarkTheme.BG_MEDIUM,
            activeforeground=DarkTheme.TEXT
        )
        cache_check.pack(side=tk.LEFT)
        
        # Save button for advanced settings
        save_button = tk.Button(
            advanced_frame,
            text="Save Settings",
            command=self.save_advanced_settings,
            bg=DarkTheme.SUCCESS,
            fg=DarkTheme.TEXT,
            font=("Arial", 11),
            pady=8,
            relief=tk.FLAT,
            activebackground=DarkTheme.SUCCESS,
            activeforeground=DarkTheme.TEXT
        )
        save_button.pack(fill=tk.X, pady=10)
        
    def save_advanced_settings(self):
        """Save Spark config from the UI to internal state"""
        self.spark_settings = {
            "master": self.spark_master.get().strip(),
            "executor_memory": self.spark_executor_memory.get(),
            "executor_cores": self.spark_executor_cores.get(),
            "executor_instances": self.spark_executor_instances.get()
        }

        # Apply to Config
        from config import config
        config.update_spark_conf(
            master=self.spark_settings["master"],
            memory=self.spark_settings["executor_memory"],
            cores=self.spark_settings["executor_cores"],
            instances=self.spark_settings["executor_instances"]
        )

        messagebox.showinfo("Settings Saved", "Advanced settings have been saved successfully.")
        self.notebook.select(0)
        
    def on_mode_change(self, event):
        """Handle mode change event"""
        mode = self.detection_mode.get()
        
        # Switch to advanced settings tab if Spark mode is selected
        if mode == "Spark Distributed":
            self.notebook.select(1)  # Select the Advanced Settings tab
            if not SPARK_AVAILABLE:
                messagebox.showwarning(
                    "PySpark Required", 
                    "Spark Distributed mode requires PySpark to be installed.\n\nPlease install it using:\npip install pyspark"
                )
        
    def create_status_bar(self):
        status_frame = tk.Frame(self.root, bg=DarkTheme.BG_MEDIUM, height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.progress_bar = ttk.Progressbar(
            status_frame,
            orient=tk.HORIZONTAL,
            mode='indeterminate'
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)
        
        status_label = tk.Label(
            status_frame,
            textvariable=self.status_text,
            bg=DarkTheme.BG_MEDIUM,
            fg=DarkTheme.TEXT,
            padx=10
        )
        status_label.pack(side=tk.RIGHT, pady=5)
        
    def select_reference_images(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png")]
        files = filedialog.askopenfilenames(title="Select Reference Images", filetypes=filetypes)
        
        if files:
            self.ref_files = files
            if len(files) == 1:
                self.ref_label.config(text=os.path.basename(files[0]))
            else:
                self.ref_label.config(text=f"{len(files)} images selected")
            
            # Show preview of first image
            try:
                img = Image.open(files[0])
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)
                self.preview_label.config(image=photo, text="")
                self.preview_label.image = photo  # Keep a reference
            except Exception as e:
                print(f"Error loading preview: {e}")
    
    def select_video_files(self):
        filetypes = [("Video files", "*.mp4 *.avi *.mov")]
        files = filedialog.askopenfilenames(title="Select Video Files", filetypes=filetypes)
        
        if files:
            self.video_files = files
            if len(files) == 1:
                self.video_label.config(text=os.path.basename(files[0]))
            else:
                self.video_label.config(text=f"{len(files)} videos selected")
    
    def run_detection(self):
        # Validation
        mode = self.detection_mode.get()
        
        if mode in ["Full Pipeline", "Missing Person Only"] and not self.ref_files:
            messagebox.showerror("Error", "Please select reference images for missing person detection.")
            return
            
        if not self.video_files:
            messagebox.showerror("Error", "Please select video files to analyze.")
            return
        
        if self.running:
            messagebox.showinfo("Info", "Detection is already running.")
            return
        
        # Check for Spark configuration if using distributed mode
        if mode == "Spark Distributed" and not SPARK_AVAILABLE:
            messagebox.showerror("Error", "PySpark is not installed. Please install it to use distributed processing.")
            return

        # Save the current settings to persist across mode changes
        self.last_used_ref_files = self.ref_files
        self.last_used_video_files = self.video_files

        # Start detection in a separate thread
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.progress_bar.start()
        self.status_text.set("Processing...")
        
        detection_thread = threading.Thread(target=self.execute_detection)
        detection_thread.daemon = True
        detection_thread.start()
    
    def execute_detection(self):
        try:
            self.start_time = time.time()  # Record the start time
            self.num_detections = 0  # Reset the detection counter

            mode = self.detection_mode.get()
            threshold = self.detection_threshold.get()
            frame_interval = self.frame_interval.get()
            use_gpu = self.use_gpu.get()
            display_video = self.display_video.get()
        
            # Set device based on GPU selection
            device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            

            if mode == "Spark Distributed" and SPARK_AVAILABLE:
                spark = initialize_spark()
                try:
                    # Initial run for missing person detection
                    missing_detections, _, ref_filenames = run_distributed_pipeline(
                        spark, self.video_files, self.ref_files, run_violence=False
                    )
                    
                    output_dir = "Output"
                    # Generate missing person report
                    if missing_detections:
                        pdf_path = os.path.join(output_dir, "missing_person_report.pdf")
                        export_to_pdf(missing_detections, ref_filenames=ref_filenames, pdf_filename=pdf_path)
                    
                    # Prompt for violence detection
                    self.root.after(0, self.prompt_violence_detection, missing_detections)
                    
                except Exception as e:
                    self.root.after(0, self.detection_error, str(e))
                finally:
                    spark.stop()
                return

        
            # Initialize detection models first
            self.root.after(0, lambda: self.status_text.set("Loading models..."))
            device, mtcnn, resnet = setup_missing_person_detection()

            output_dir = "Output"
            os.makedirs(output_dir, exist_ok=True)

            if mode in ["Full Pipeline", "Missing Person Only"]:
                # Process reference images
                ref_embeddings = []
                self.root.after(0, lambda: self.status_text.set("Processing reference images..."))
            
                for ref_filename in self.ref_files:
                    ref_img = Image.open(ref_filename).convert("RGB")
                    faces, probs = mtcnn(ref_img, return_prob=True)
                    if faces is None or (hasattr(faces, '__len__') and len(faces) == 0):
                        continue
                    
                    # Use highest probability face if multiple are detected
                    ref_face = faces[int(np.argmax(probs))] if faces.ndim == 4 else faces
                    with torch.no_grad():
                        emb = resnet(
                            ref_face.unsqueeze(0).to(device).half() if device.type=='cuda'
                            else ref_face.unsqueeze(0).to(device)
                        )
                    ref_embeddings.append(emb)
            
                if not ref_embeddings:
                    raise Exception("No valid faces detected in the reference images.")
                
                # Convert reference embeddings to half precision if on GPU
                if device.type == 'cuda':
                    ref_embeddings = [emb.half() for emb in ref_embeddings]
                
                # Process each video
                all_detections = []
                for video_file in self.video_files:
                    self.root.after(0, lambda v=video_file: self.status_text.set(f"Processing video: {os.path.basename(v)}..."))
                
                    # Call process_video function with parameters
                    detections = process_video(
                        video_file, 
                        mtcnn, 
                        resnet, 
                        device, 
                        ref_embeddings, 
                        frame_interval=frame_interval,
                        detection_threshold=threshold
                        # display_video=self.display_video.get()  # Pass the flag
                    )
                    all_detections.extend(detections)
                    self.num_detections += len(detections)  # Update the detection count
                
                # Export results    
                if all_detections:
                    self.root.after(0, lambda: self.status_text.set("Generating report..."))
                    all_detections.sort(key=lambda x: x['similarity'], reverse=True)
                    export_to_pdf(all_detections, ref_filenames=self.ref_files)
                
            if mode in ["Full Pipeline", "Violence Only"]:
                # Violence detection part
                self.root.after(0, lambda: self.status_text.set("Detecting violence..."))
            
                # Load violence model
                model = load_violence_detection_model(device)
            
                # Store all violence detections for potential combined report
                all_violence_detections = []

                # Process each video
                for video_file in self.video_files:
                    self.root.after(0, lambda v=video_file: self.status_text.set(f"Checking violence in: {os.path.basename(v)}..."))
                    violence_detections = detect_violence_in_video(
                        video_file, 
                        model, 
                        device, 
                        threshold=threshold
                        # display_video=self.display_video.get()  # Pass the flag
                    )
                    self.num_detections += len(violence_detections)  # Update the detection count
                    all_violence_detections.extend(violence_detections)
                
                    # if violence_detections:
                    #     # Generate individual report for this video
                    #     video_name = os.path.splitext(os.path.basename(video_file))[0]
                    #     pdf_path = os.path.join(output_dir, f"violence_{video_name}.pdf")
                    #     export_violence_report(violence_detections, video_file, pdf_filename=pdf_path)
                
                # Generate a combined violence report if needed
                if all_violence_detections and len(self.video_files) > 1:
                    combined_violence_path = os.path.join(output_dir, "violence_combined_report.pdf")
                    export_violence_report(all_violence_detections, "Multiple Videos", pdf_filename=combined_violence_path)
                    # self.root.after(0, self.detection_complete)


            # If both types of detections were run, consider generating a combined report
            if mode == "Full Pipeline" and (all_detections or all_violence_detections):
                combined_path = os.path.join(output_dir, "combined_analysis_report.pdf")
                generate_combined_report(all_detections if 'all_detections' in locals() else [], 
                                    all_violence_detections if 'all_violence_detections' in locals() else [], 
                                    self.ref_files, output_dir)        
            self.root.after(0, self.detection_complete)
            
        except Exception as e:
            err_msg=str(e)
            self.root.after(0, lambda: self.detection_error(err_msg))
    
    def prompt_violence_detection(self, missing_detections):
        """Prompt user to run violence detection after missing person results"""
        if missing_detections:
            msg = "Missing persons detected. Run violence detection on these videos?"
        else:
            msg = "Run violence detection on all videos?"
        
        response = messagebox.askyesno("Violence Detection", msg)
        if response:
            # Determine target videos
            target_videos = list(set(d['video_path'] for d in missing_detections)) if missing_detections else self.video_files
            
            # Run violence detection in a new thread
            violence_thread = threading.Thread(target=self.run_violence_detection, args=(target_videos,))
            violence_thread.daemon = True
            violence_thread.start()
        else:
            self.root.after(0, self.detection_complete)
    
    def run_violence_detection(self, target_videos):
        """Run violence detection on selected videos"""
        try:
            spark = initialize_spark()
            try:
                # Run violence detection
                _, violence_detections, _ = run_distributed_pipeline(
                    spark, target_videos, run_violence=True
                )
                
                # Generate reports
                if violence_detections:
                    output_dir = "Output"
                    from report_generation import export_violence_report, generate_combined_report
                    
                    # Violence report
                    pdf_path = os.path.join(output_dir, "violence_report.pdf")
                    export_violence_report(violence_detections, "Violence Detection", pdf_filename=pdf_path)
                    
                    # Combined report
                    combined_path = os.path.join(output_dir, "combined_report.pdf")
                    generate_combined_report([], violence_detections, [], output_dir)
                    
                    self.root.after(0, self.detection_complete)
                else:
                    self.root.after(0, self.detection_complete)
            except Exception as e:
                self.root.after(0, self.detection_error, str(e))
            finally:
                spark.stop()
        except Exception as e:
            self.root.after(0, self.detection_error, str(e))

    def detection_complete(self):
        self.progress_bar.stop()
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.status_text.set("Detection completed")

         # Calculate the time taken
        time_taken = time.time() - self.start_time
        time_taken_str = f"{time_taken:.2f} seconds"
    
        # Check if PDF files were generated
        pdf_files = []
        output_dir = "Output"
        for file in os.listdir(output_dir):
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(output_dir, file))
    
        if pdf_files:
            # Create a dialog with buttons to open the PDFs
            result = messagebox.askquestion("Detection Complete", 
                                        f"Detection process completed successfully.\nTime taken: {time_taken_str}\nNumber of detections: {self.num_detections}\nWould you like to view the results?")
            if result == "yes":
                self.show_pdf_viewer(pdf_files)
        else:
            messagebox.showinfo("Complete", f"Detection process completed successfully.\nTime taken: {time_taken_str}\nNumber of detections: {self.num_detections}")

    def show_pdf_viewer(self, pdf_files):
        """Display a window with buttons to open available PDF reports"""
        pdf_window = tk.Toplevel(self.root)
        pdf_window.title("Detection Reports")
        pdf_window.geometry("400x300")
        pdf_window.configure(bg="#f5f5f5")
    
        # Header
        tk.Label(
            pdf_window, 
            text="Available Detection Reports",
            font=("Arial", 14, "bold"),
            bg="#f5f5f5",
            pady=10
        ).pack(fill=tk.X)
    
        # Create a frame for the PDF list
        list_frame = tk.Frame(pdf_window, bg="#f5f5f5", padx=20, pady=10)
        list_frame.pack(fill=tk.BOTH, expand=True)
    
        # Add a button for each PDF file
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
        
            # Determine file type for icon/color
            if "violence" in filename.lower():
                bg_color = "#e74c3c"  # Red for violence
                prefix = "🔍 "
            else:
                bg_color = "#3498db"  # Blue for missing person
                prefix = "👤 "
            
            button_frame = tk.Frame(list_frame, bg="#f5f5f5", pady=5)
            button_frame.pack(fill=tk.X)
        
            # Button to open the PDF
            open_button = tk.Button(
                button_frame,
                text=f"{prefix} Open {filename}",
                command=lambda f=pdf_file: self.open_pdf(f),
                bg=bg_color,
                fg="white",
                font=("Arial", 11),
                pady=8
            )
            open_button.pack(fill=tk.X)
    
        # Close button
        tk.Button(
            pdf_window,
            text="Close",
            command=pdf_window.destroy,
            bg="#7f8c8d",
            fg="white",
            font=("Arial", 11),
            pady=5
        ).pack(fill=tk.X, padx=20, pady=10)

    def open_pdf(self,pdf_path):
        """Open the PDF file with the default viewer"""
        try:
            import platform
            import subprocess
        
            if platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', pdf_path))
            elif platform.system() == 'Windows':  # Windows
                os.startfile(pdf_path)
            else:  # Linux
                subprocess.call(('xdg-open', pdf_path))
        except Exception as e:
            messagebox.showerror("Error", f"Could not open PDF: {e}")
    
    def detection_error(self, error_msg):
        self.progress_bar.stop()
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.status_text.set("Error")
        messagebox.showerror("Error", f"An error occurred during detection:\n{error_msg}")

    # Add to MissingPersonDetectionApp class in new_ui.py
    def create_performance_panel(self, parent):
        """Create performance metrics display panel"""
        perf_frame = tk.Frame(parent, bg=DarkTheme.BG_MEDIUM)
        perf_frame.pack(fill=tk.X, pady=10)
    
        # Metrics labels
        self.metrics_vars = {
            'accuracy': tk.StringVar(value="Accuracy: -"),
            'precision': tk.StringVar(value="Precision: -"),
            'recall': tk.StringVar(value="Recall: -"),
            'fps': tk.StringVar(value="FPS: -"),
            # 'total_time': tk.StringVar(value="Total Time: -"),
            'detections': tk.StringVar(value="Detections: 0")
        }
    
        for metric, var in self.metrics_vars.items():
            label = tk.Label(
                perf_frame,
                textvariable=var,
                bg=DarkTheme.BG_MEDIUM,
                fg=DarkTheme.TEXT,
                font=("Arial", 9),
                anchor=tk.W
            )
            label.pack(fill=tk.X, padx=5, pady=2)
    
        # # Performance recommendations
        # self.recommendation_var = tk.StringVar(value="Recommendations: -")
        
        rec_label = tk.Label(
            perf_frame,
            textvariable=self.recommendation_var,
            bg=DarkTheme.BG_MEDIUM,
            fg=DarkTheme.ACCENT,
            font=("Arial", 9, "italic"),
            anchor=tk.W,
            wraplength=350
        )
        rec_label.pack(fill=tk.X, padx=5, pady=5)
    
        # Start periodic updates
        self.update_performance_metrics()

    def update_performance_metrics(self):
        """Periodically update performance metrics display"""
        if self.running:
            try:
                metrics = stats_monitor.get_performance_metrics()
                perf_stats = calculate_performance_stats()

                # Update UI elements safely with default values
                self.metrics_vars['accuracy'].set(
                f"Accuracy: {metrics.get('accuracy', 0)*100:.1f}%"
                )
                self.metrics_vars['precision'].set(
                f"Precision: {metrics.get('precision', 0)*100:.1f}%"
                )
                self.metrics_vars['recall'].set(
                f"Recall: {metrics.get('recall', 0)*100:.1f}%"
                )
                self.metrics_vars['fps'].set(
                f"FPS: {metrics.get('fps', 0):.1f}"
                )
                self.metrics_vars['detections'].set(
                f"Detections: TP={metrics.get('true_positives', 0)} FP={metrics.get('false_positives', 0)}"
                )
                # self.metrics_vars['total_time'].set(
                # f"Total Time: {perf_stats.get('total_processing_time', 0):.1f}s"
                # )
                # self.metrics_vars['total_frames'].set(
                # f"Frames: {perf_stats.get('frames_processed', 0)}"
                # )

                # Generate recommendations
                recommendations = []
                current_fps = metrics.get('fps', 0)
                current_precision = metrics.get('precision', 0)
                current_recall = metrics.get('recall', 0)
            
                if current_fps < 10:
                    recommendations.append("Consider increasing frame interval")
                if current_precision < 0.7:
                    recommendations.append("Increase detection threshold")
                if current_recall < 0.7:
                    recommendations.append("Decrease detection threshold")
                
                self.recommendation_var.set(
                    "Recommendations: " + ("; ".join(recommendations) if recommendations else "No recommendations")
                )
            
            except Exception as e:
                print(f"Error updating metrics: {str(e)}")
                # Optionally log full error for debugging:
                # import traceback
                # traceback.print_exc()

    
        # Schedule next update
        self.root.after(2000, self.update_performance_metrics)

# Main application
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("Output", exist_ok=True)
    
    # Initialize the app
    root = tk.Tk()
    app = MissingPersonDetectionApp(root)
    
    # Set window icon (if available)
    try:
        root.iconbitmap("icon.ico")  # Replace with your icon path
    except:
        pass
        
    root.mainloop()