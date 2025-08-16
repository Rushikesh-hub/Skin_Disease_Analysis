
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkFont
from PIL import Image, ImageTk, ImageOps
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input


# Configure TensorFlow for GPU
def configure_tensorflow():
    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured {len(gpus)} GPU(s) for dynamic memory allocation")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")


configure_tensorflow()

# Default class names (can be updated when model is loaded)
CLASS_NAMES = [
    "Eczema",
    "Viral Infections",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Melanocytic Nevi",
    "Keratosis-like Lesions",
    "Psoriasis & Lichen Planus",
    "Seborrheic Keratoses",
    "Fungal Infections"
]


class SkinDiseasePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Skin Disease Detection System - Inference Only")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a2e')

        # Initialize variables
        self.model = None
        self.training_history = None
        self.model_path = "my_model.h5"
        self.history_path = "history.json"
        self.current_image = None
        self.current_image_path = ""

        # Color scheme
        self.colors = {
            'bg_primary': '#1a1a2e',
            'bg_secondary': '#16213e',
            'bg_tertiary': '#0f3460',
            'accent': '#e94560',
            'accent_light': '#f39c12',
            'text_primary': '#ffffff',
            'text_secondary': '#bdc3c7',
            'success': '#27ae60',
            'warning': '#f39c12',
            'error': '#e74c3c'
        }

        self.setup_styles()
        self.create_widgets()
        self.load_saved_model()

    def setup_styles(self):
        """Configure custom styles for the application"""
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Configure styles
        self.style.configure('Title.TLabel',
                             background=self.colors['bg_primary'],
                             foreground=self.colors['text_primary'],
                             font=('Arial', 24, 'bold'))

        self.style.configure('Heading.TLabel',
                             background=self.colors['bg_secondary'],
                             foreground=self.colors['accent'],
                             font=('Arial', 14, 'bold'))

        self.style.configure('Info.TLabel',
                             background=self.colors['bg_secondary'],
                             foreground=self.colors['text_secondary'],
                             font=('Arial', 10))

        self.style.configure('Custom.TButton',
                             background=self.colors['accent'],
                             foreground=self.colors['text_primary'],
                             font=('Arial', 11, 'bold'),
                             borderwidth=0)

        self.style.configure('Success.TButton',
                             background=self.colors['success'],
                             foreground=self.colors['text_primary'],
                             font=('Arial', 11, 'bold'))

        self.style.configure('Warning.TButton',
                             background=self.colors['warning'],
                             foreground=self.colors['text_primary'],
                             font=('Arial', 11, 'bold'))

    def create_widgets(self):
        """Create and layout all GUI components"""
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Title
        title_frame = tk.Frame(main_container, bg=self.colors['bg_primary'])
        title_frame.pack(fill='x', pady=(0, 20))

        title_label = ttk.Label(title_frame,
                                text="🔬 Skin Disease Detection System",
                                style='Title.TLabel')
        title_label.pack()

        subtitle_label = ttk.Label(title_frame,
                                   text="Inference Only - Load Pre-trained Model for Predictions",
                                   style='Info.TLabel')
        subtitle_label.pack()

        # Main content area with notebook
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True)

        # Create tabs
        self.create_prediction_tab()
        self.create_model_management_tab()
        self.create_evaluation_tab()
        self.create_model_info_tab()

        # Status bar
        self.create_status_bar(main_container)

    def create_prediction_tab(self):
        """Create prediction tab for image analysis"""
        prediction_frame = tk.Frame(self.notebook, bg=self.colors['bg_secondary'])
        self.notebook.add(prediction_frame, text="🔍 Disease Prediction")

        # Left panel for image
        left_panel = tk.Frame(prediction_frame, bg=self.colors['bg_secondary'])
        left_panel.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        # Image upload section
        image_frame = tk.LabelFrame(left_panel, text="Upload Image for Analysis",
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text_primary'],
                                    font=('Arial', 12, 'bold'))
        image_frame.pack(fill='x', pady=(0, 10))

        # Upload button
        upload_btn = ttk.Button(image_frame, text="📁 Select Image",
                                command=self.upload_image,
                                style='Custom.TButton')
        upload_btn.pack(pady=10)

        # Supported formats info
        formats_label = tk.Label(image_frame,
                                 text="Supported: JPG, PNG, BMP, GIF, TIFF",
                                 bg=self.colors['bg_secondary'],
                                 fg=self.colors['text_secondary'],
                                 font=('Arial', 9))
        formats_label.pack()

        # Image display
        self.image_label = tk.Label(left_panel, bg=self.colors['bg_tertiary'],
                                    text="No image selected\n\nClick 'Select Image' to upload",
                                    fg=self.colors['text_secondary'],
                                    font=('Arial', 12), width=40, height=20)
        self.image_label.pack(fill='both', expand=True, pady=(10, 0))

        # Right panel for results
        right_panel = tk.Frame(prediction_frame, bg=self.colors['bg_secondary'])
        right_panel.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        # Prediction controls
        control_frame = tk.LabelFrame(right_panel, text="Analysis Controls",
                                      bg=self.colors['bg_secondary'],
                                      fg=self.colors['text_primary'],
                                      font=('Arial', 12, 'bold'))
        control_frame.pack(fill='x', pady=(0, 10))

        predict_btn = ttk.Button(control_frame, text="🔬 Analyze Image",
                                 command=self.predict_disease,
                                 style='Success.TButton')
        predict_btn.pack(pady=10)

        clear_btn = ttk.Button(control_frame, text="🗑️ Clear Results",
                               command=self.clear_results,
                               style='Warning.TButton')
        clear_btn.pack(pady=5)

        # Results section
        results_frame = tk.LabelFrame(right_panel, text="Analysis Results",
                                      bg=self.colors['bg_secondary'],
                                      fg=self.colors['text_primary'],
                                      font=('Arial', 12, 'bold'))
        results_frame.pack(fill='both', expand=True)

        # Primary prediction
        tk.Label(results_frame, text="Primary Diagnosis:",
                 bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                 font=('Arial', 11, 'bold')).pack(anchor='w', padx=10, pady=(10, 2))

        self.primary_result = tk.Label(results_frame, text="No prediction yet",
                                       bg=self.colors['bg_secondary'],
                                       fg=self.colors['accent'],
                                       font=('Arial', 16, 'bold'))
        self.primary_result.pack(anchor='w', padx=10, pady=(0, 10))

        # Confidence
        tk.Label(results_frame, text="Confidence Level:",
                 bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                 font=('Arial', 11, 'bold')).pack(anchor='w', padx=10, pady=(0, 2))

        self.confidence_result = tk.Label(results_frame, text="0.0%",
                                          bg=self.colors['bg_secondary'],
                                          fg=self.colors['success'],
                                          font=('Arial', 14, 'bold'))
        self.confidence_result.pack(anchor='w', padx=10, pady=(0, 10))

        # All predictions
        tk.Label(results_frame, text="Detailed Predictions:",
                 bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                 font=('Arial', 11, 'bold')).pack(anchor='w', padx=10, pady=(0, 5))

        # Frame for results with scrollbar
        results_text_frame = tk.Frame(results_frame, bg=self.colors['bg_secondary'])
        results_text_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        self.all_results = tk.Text(results_text_frame, height=8,
                                   bg=self.colors['bg_tertiary'],
                                   fg=self.colors['text_primary'],
                                   font=('Arial', 10))

        results_scrollbar = tk.Scrollbar(results_text_frame)
        results_scrollbar.pack(side='right', fill='y')
        self.all_results.pack(side='left', fill='both', expand=True)

        self.all_results.config(yscrollcommand=results_scrollbar.set)
        results_scrollbar.config(command=self.all_results.yview)

    def create_model_management_tab(self):
        """Create model management tab"""
        model_frame = tk.Frame(self.notebook, bg=self.colors['bg_secondary'])
        self.notebook.add(model_frame, text="🤖 Model Management")

        # Load model section
        load_frame = tk.LabelFrame(model_frame, text="Load Pre-trained Model",
                                   bg=self.colors['bg_secondary'],
                                   fg=self.colors['text_primary'],
                                   font=('Arial', 12, 'bold'))
        load_frame.pack(fill='x', padx=20, pady=10)

        # Model path selection
        tk.Label(load_frame, text="Model File (.h5 or .keras):",
                 bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                 font=('Arial', 11)).pack(anchor='w', padx=10, pady=(10, 5))

        model_path_frame = tk.Frame(load_frame, bg=self.colors['bg_secondary'])
        model_path_frame.pack(fill='x', padx=10, pady=5)

        self.model_path_var = tk.StringVar(value=self.model_path)
        self.model_path_entry = tk.Entry(model_path_frame,
                                         textvariable=self.model_path_var,
                                         font=('Arial', 10), width=50)
        self.model_path_entry.pack(side='left', fill='x', expand=True)

        browse_model_btn = ttk.Button(model_path_frame, text="Browse",
                                      command=self.browse_model,
                                      style='Custom.TButton')
        browse_model_btn.pack(side='right', padx=(10, 0))

        # Load button
        load_model_btn = ttk.Button(load_frame, text="📂 Load Model",
                                    command=self.load_model_from_path,
                                    style='Success.TButton')
        load_model_btn.pack(pady=10)

        # Custom class names section
        class_frame = tk.LabelFrame(model_frame, text="Custom Class Names (Optional)",
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text_primary'],
                                    font=('Arial', 12, 'bold'))
        class_frame.pack(fill='both', expand=True, padx=20, pady=10)

        tk.Label(class_frame, text="If your model uses different class names, enter them here (one per line):",
                 bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'],
                 font=('Arial', 10)).pack(anchor='w', padx=10, pady=(10, 5))

        # Text area for class names
        class_text_frame = tk.Frame(class_frame, bg=self.colors['bg_secondary'])
        class_text_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.class_names_text = tk.Text(class_text_frame, height=8,
                                        bg=self.colors['bg_tertiary'],
                                        fg=self.colors['text_primary'],
                                        font=('Arial', 10))

        class_scrollbar = tk.Scrollbar(class_text_frame)
        class_scrollbar.pack(side='right', fill='y')
        self.class_names_text.pack(side='left', fill='both', expand=True)

        self.class_names_text.config(yscrollcommand=class_scrollbar.set)
        class_scrollbar.config(command=self.class_names_text.yview)

        # Load default class names
        self.class_names_text.insert('1.0', '\n'.join(CLASS_NAMES))

        # Update class names button
        update_classes_btn = ttk.Button(class_frame, text="📝 Update Class Names",
                                        command=self.update_class_names,
                                        style='Warning.TButton')
        update_classes_btn.pack(pady=10)

    def create_evaluation_tab(self):
        """Create model evaluation tab with training history visualization"""
        eval_frame = tk.Frame(self.notebook, bg=self.colors['bg_secondary'])
        self.notebook.add(eval_frame, text="📊 Training History")

        # Control panel
        control_frame = tk.Frame(eval_frame, bg=self.colors['bg_secondary'])
        control_frame.pack(fill='x', padx=20, pady=10)

        # Load history file
        history_path_frame = tk.Frame(control_frame, bg=self.colors['bg_secondary'])
        history_path_frame.pack(fill='x', pady=5)

        tk.Label(history_path_frame, text="Training History File (JSON):",
                 bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                 font=('Arial', 10)).pack(side='left', padx=(0, 10))

        self.history_path_var = tk.StringVar(value=self.history_path)
        history_entry = tk.Entry(history_path_frame,
                                 textvariable=self.history_path_var,
                                 font=('Arial', 9), width=40)
        history_entry.pack(side='left', fill='x', expand=True)

        browse_history_btn = ttk.Button(history_path_frame, text="Browse",
                                        command=self.browse_history,
                                        style='Custom.TButton')
        browse_history_btn.pack(side='right', padx=(10, 0))

        # Control buttons
        button_frame = tk.Frame(control_frame, bg=self.colors['bg_secondary'])
        button_frame.pack(fill='x', pady=10)

        load_history_btn = ttk.Button(button_frame, text="📈 Load Training History",
                                      command=self.load_training_history,
                                      style='Success.TButton')
        load_history_btn.pack(side='left', padx=(0, 10))

        plot_btn = ttk.Button(button_frame, text="📊 Generate Plots",
                              command=self.generate_evaluations,
                              style='Custom.TButton')
        plot_btn.pack(side='left')

        # Plot area
        self.plot_frame = tk.Frame(eval_frame, bg=self.colors['bg_secondary'])
        self.plot_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Default message
        self.default_plot_label = tk.Label(self.plot_frame,
                                           text="Load training history and click 'Generate Plots' to view training metrics",
                                           bg=self.colors['bg_secondary'],
                                           fg=self.colors['text_secondary'],
                                           font=('Arial', 12))
        self.default_plot_label.pack(expand=True)

    def create_model_info_tab(self):
        """Create model information and settings tab"""
        info_frame = tk.Frame(self.notebook, bg=self.colors['bg_secondary'])
        self.notebook.add(info_frame, text="ℹ️ Model Info")

        # Model status
        status_frame = tk.LabelFrame(info_frame, text="Model Status",
                                     bg=self.colors['bg_secondary'],
                                     fg=self.colors['text_primary'],
                                     font=('Arial', 12, 'bold'))
        status_frame.pack(fill='x', padx=20, pady=10)

        self.model_status = tk.Text(status_frame, height=10,
                                    bg=self.colors['bg_tertiary'],
                                    fg=self.colors['text_primary'],
                                    font=('Consolas', 10))
        self.model_status.pack(fill='x', padx=10, pady=10)

        # System info
        system_frame = tk.LabelFrame(info_frame, text="System Information",
                                     bg=self.colors['bg_secondary'],
                                     fg=self.colors['text_primary'],
                                     font=('Arial', 12, 'bold'))
        system_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.system_info = tk.Text(system_frame, bg=self.colors['bg_tertiary'],
                                   fg=self.colors['text_secondary'],
                                   font=('Consolas', 9))
        self.system_info.pack(fill='both', expand=True, padx=10, pady=10)

        self.update_system_info()

    def create_status_bar(self, parent):
        """Create status bar at the bottom"""
        status_frame = tk.Frame(parent, bg=self.colors['bg_tertiary'], height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load a pre-trained model to begin predictions")

        status_label = tk.Label(status_frame, textvariable=self.status_var,
                                bg=self.colors['bg_tertiary'],
                                fg=self.colors['text_secondary'],
                                font=('Arial', 9), anchor='w')
        status_label.pack(fill='both', padx=10, pady=5)

    def browse_model(self):
        """Browse for model file"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model files", "*.h5 *.keras"), ("All files", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
            self.model_path = file_path

    def browse_history(self):
        """Browse for training history file"""
        file_path = filedialog.askopenfilename(
            title="Select Training History File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.history_path_var.set(file_path)
            self.history_path = file_path

    def load_model_from_path(self):
        """Load model from specified path"""
        model_path = self.model_path_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file!")
            return

        try:
            self.update_status("Loading model...")
            self.model = keras.models.load_model(model_path)
            self.model_path = model_path
            self.update_status("Model loaded successfully!")
            self.update_model_status()
            messagebox.showinfo("Success", f"Model loaded successfully!\nFile: {os.path.basename(model_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.update_status(f"Failed to load model: {str(e)}")

    def load_saved_model(self):
        """Load previously saved model"""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                self.update_status("Model loaded successfully")
                self.update_model_status()

                # Load training history if available
                if os.path.exists(self.history_path):
                    self.load_training_history()
            else:
                self.update_status("No saved model found - Please load a pre-trained model")
        except Exception as e:
            self.update_status(f"Failed to load model: {str(e)}")

    def load_training_history(self):
        """Load training history from JSON file"""
        history_path = self.history_path_var.get()
        if not history_path or not os.path.exists(history_path):
            messagebox.showerror("Error", "Please select a valid history file!")
            return

        try:
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
            self.update_status("Training history loaded successfully")
            messagebox.showinfo("Success", "Training history loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load training history: {str(e)}")

    def update_class_names(self):
        """Update class names from text area"""
        try:
            class_text = self.class_names_text.get('1.0', tk.END).strip()
            new_class_names = [line.strip() for line in class_text.split('\n') if line.strip()]

            if len(new_class_names) == 0:
                messagebox.showwarning("Warning", "Please enter at least one class name!")
                return

            global CLASS_NAMES
            CLASS_NAMES = new_class_names
            self.update_status(f"Updated class names - {len(CLASS_NAMES)} classes")
            self.update_model_status()
            messagebox.showinfo("Success", f"Class names updated! ({len(CLASS_NAMES)} classes)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update class names: {str(e)}")

    def upload_image(self):
        """Upload image for prediction"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )

        if file_path:
            try:
                # Load and display image
                image = Image.open(file_path)
                self.current_image = image
                self.current_image_path = file_path

                # Resize for display
                display_size = (300, 300)
                image_display = ImageOps.fit(image, display_size, Image.Resampling.LANCZOS)

                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image_display)
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo  # Keep a reference

                self.update_status(f"Image loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def predict_disease(self):
        """Predict disease from uploaded image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return

        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first!")
            return

        try:
            self.update_status("Analyzing image...")

            # Preprocess image for ResNet
            img = self.current_image.resize((224, 224))
            img_array = np.array(img, dtype=np.float32)

            # Apply ResNet preprocessing
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)[0]

            # Ensure we have the right number of classes
            if len(predictions) != len(CLASS_NAMES):
                messagebox.showwarning("Warning",
                                       f"Model output ({len(predictions)} classes) doesn't match class names ({len(CLASS_NAMES)} classes)!\n"
                                       f"Please update class names in Model Management tab.")
                return

            # Get results
            results = [(CLASS_NAMES[i], predictions[i] * 100) for i in range(len(CLASS_NAMES))]
            results.sort(key=lambda x: x[1], reverse=True)

            # Update UI
            self.primary_result.config(text=results[0][0])
            self.confidence_result.config(text=f"{results[0][1]:.1f}%")

            # Color code confidence
            confidence = results[0][1]
            if confidence >= 80:
                self.confidence_result.config(fg=self.colors['success'])
            elif confidence >= 60:
                self.confidence_result.config(fg=self.colors['warning'])
            else:
                self.confidence_result.config(fg=self.colors['error'])

            # Show all predictions
            self.all_results.delete(1.0, tk.END)
            self.all_results.insert(tk.END, f"Analysis Results for: {os.path.basename(self.current_image_path)}\n")
            self.all_results.insert(tk.END, "=" * 50 + "\n\n")

            for i, (disease, confidence) in enumerate(results, 1):
                confidence_bar = "█" * int(confidence / 5) + "░" * (20 - int(confidence / 5))
                self.all_results.insert(tk.END, f"{i:2d}. {disease:<25} {confidence:5.1f}% [{confidence_bar}]\n")

            # Add interpretation
            self.all_results.insert(tk.END, "\n" + "=" * 50 + "\n")
            if results[0][1] >= 80:
                self.all_results.insert(tk.END, "🟢 High Confidence: Strong indication of the predicted condition\n")
            elif results[0][1] >= 60:
                self.all_results.insert(tk.END,
                                        "🟡 Moderate Confidence: Consider consulting a healthcare professional\n")
            else:
                self.all_results.insert(tk.END, "🔴 Low Confidence: Seek professional medical evaluation\n")

            self.all_results.insert(tk.END,
                                    "\n⚠️  DISCLAIMER: This is an AI prediction tool and should not replace professional medical diagnosis.")

            self.update_status(f"Analysis completed - {results[0][0]} ({results[0][1]:.1f}%)")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.update_status(f"Prediction failed: {str(e)}")

    def clear_results(self):
        """Clear prediction results"""
        self.primary_result.config(text="No prediction yet")
        self.confidence_result.config(text="0.0%", fg=self.colors['success'])
        self.all_results.delete(1.0, tk.END)
        self.update_status("Results cleared")

    def generate_evaluations(self):
        """Generate evaluation plots from training history"""
        if self.training_history is None:
            messagebox.showwarning("Warning", "Please load training history first!")
            return

        try:
            # Clear previous plots
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            # Create training history plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.patch.set_facecolor(self.colors['bg_secondary'])

            # Configure all axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_facecolor(self.colors['bg_tertiary'])
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_color('white')

            # Plot 1: Loss
            if 'loss' in self.training_history and 'val_loss' in self.training_history:
                epochs = range(1, len(self.training_history['loss']) + 1)
                ax1.plot(epochs, self.training_history['loss'], 'o-', color='#e74c3c', label='Train Loss', linewidth=2)
                ax1.plot(epochs, self.training_history['val_loss'], 'o-', color='#3498db', label='Val Loss',
                         linewidth=2)
                ax1.set_title('Training & Validation Loss', color='white', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Epoch', color='white')
                ax1.set_ylabel('Loss', color='white')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # Plot 2: Accuracy
            if 'accuracy' in self.training_history and 'val_accuracy' in self.training_history:
                ax2.plot(epochs, self.training_history['accuracy'], 'o-', color='#27ae60', label='Train Acc',
                         linewidth=2)
                ax2.plot(epochs, self.training_history['val_accuracy'], 'o-', color='#f39c12', label='Val Acc',
                         linewidth=2)
                ax2.set_title('Training & Validation Accuracy', color='white', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Epoch', color='white')
                ax2.set_ylabel('Accuracy', color='white')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # Plot 3: Learning Rate (if available)
            if 'lr' in self.training_history:
                ax3.plot(epochs, self.training_history['lr'], 'o-', color='#9b59b6', linewidth=2)
                ax3.set_title('Learning Rate Schedule', color='white', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Epoch', color='white')
                ax3.set_ylabel('Learning Rate', color='white')
                ax3.set_yscale('log')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Learning Rate\nData Not Available',
                         transform=ax3.transAxes, ha='center', va='center',
                         color='white', fontsize=12)
                ax3.set_title('Learning Rate Schedule', color='white', fontsize=12, fontweight='bold')

            # Plot 4: Loss difference (overfitting indicator)
            if 'loss' in self.training_history and 'val_loss' in self.training_history:
                loss_diff = [abs(t - v) for t, v in
                             zip(self.training_history['loss'], self.training_history['val_loss'])]
                ax4.plot(epochs, loss_diff, 'o-', color='#e67e22', label='|Train - Val| Loss', linewidth=2)
                ax4.set_title('Overfitting Monitor', color='white', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Epoch', color='white')
                ax4.set_ylabel('Loss Difference', color='white')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

            self.update_status("Training history plots generated successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate evaluation plots: {str(e)}")

    def update_model_status(self):
        """Update model status information"""
        if self.model:
            try:
                # Get model file size if it exists
                model_size_text = "Unknown"
                if os.path.exists(self.model_path):
                    model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
                    model_size_text = f"{model_size_mb:.2f} MB"

                # Count trainable parameters
                trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
                total_params = self.model.count_params()

                status_text = f"""Model Information:
════════════════════════════════════════════════════════════════

📂 File Information:
   Path: {self.model_path}
   Size: {model_size_text}
   Loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🏗️  Architecture:
   Model Type: {type(self.model).__name__}
   Input Shape: {self.model.input_shape}
   Output Shape: {self.model.output_shape}
   Total Layers: {len(self.model.layers)}

📊 Parameters:
   Total Parameters: {total_params:,}
   Trainable Parameters: {trainable_params:,}
   Non-trainable Parameters: {total_params - trainable_params:,}

🏷️  Classes ({len(CLASS_NAMES)}):
"""
                for i, class_name in enumerate(CLASS_NAMES, 1):
                    status_text += f"   {i:2d}. {class_name}\n"

                status_text += f"""
🔧 Model Layers (first 10):
"""
                for i, layer in enumerate(self.model.layers[:10]):
                    layer_params = layer.count_params()
                    status_text += f"   {i + 1:2d}. {layer.name} ({layer.__class__.__name__}) - {layer_params:,} params\n"

                if len(self.model.layers) > 10:
                    status_text += f"   ... and {len(self.model.layers) - 10} more layers\n"

                status_text += f"""
✅ Status: Model ready for predictions
   Compatible with 224x224 RGB images
   Preprocessing: ResNet-style normalization
"""

            except Exception as e:
                status_text = f"Error retrieving model information: {str(e)}"

            self.model_status.delete(1.0, tk.END)
            self.model_status.insert(1.0, status_text)
        else:
            self.model_status.delete(1.0, tk.END)
            self.model_status.insert(1.0, """No model loaded.

To get started:
1. Go to 'Model Management' tab
2. Select your pre-trained model file (.h5 or .keras)
3. Click 'Load Model'
4. Optionally update class names if needed
5. Return to 'Disease Prediction' tab to analyze images

Expected model format:
- Input: (None, 224, 224, 3) - RGB images
- Output: (None, num_classes) - class probabilities
- Preprocessing: ResNet-style (preprocess_input)
""")

    def update_system_info(self):
        """Update system information"""
        # GPU information
        gpu_info = "No GPU detected"
        try:
            if tf.config.list_physical_devices('GPU'):
                gpu_devices = tf.config.list_physical_devices('GPU')
                gpu_info = f"GPU(s) Available: {len(gpu_devices)}"
                for i, gpu in enumerate(gpu_devices):
                    gpu_info += f"\n   GPU {i}: {gpu.name}"
        except:
            gpu_info = "GPU detection failed"

        system_text = f"""System Information:
════════════════════════════════════════════════════════════════

🖥️  Software Environment:
   TensorFlow Version: {tf.__version__}
   CUDA Support: {tf.test.is_built_with_cuda()}
   GPU Configuration: Dynamic memory growth enabled
   {gpu_info}

🔬 Application Configuration:
   Mode: Inference Only (No Training)
   Image Input Size: 224 x 224 pixels
   Supported Formats: JPG, PNG, BMP, GIF, TIFF
   Preprocessing: ResNet-style normalization

🏷️  Default Disease Classes:
"""

        for i, disease in enumerate(CLASS_NAMES, 1):
            system_text += f"   {i:2d}. {disease}\n"

        system_text += f"""
📁 Expected File Structure:
   my_model.h5 (or .keras) - Your trained model
   history.json - Training history (optional)

💡 Usage Instructions:
   1. Load your pre-trained model
   2. Upload an image for analysis
   3. Click 'Analyze Image' for predictions
   4. View detailed results and confidence scores

⚠️  Important Notes:
   - This tool is for research/educational purposes
   - AI predictions should not replace professional medical diagnosis
   - Always consult healthcare professionals for medical concerns
   - Model accuracy depends on training data quality
"""

        self.system_info.delete(1.0, tk.END)
        self.system_info.insert(1.0, system_text)

    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
        self.root.update_idletasks()


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = SkinDiseasePredictor(root)

    # Set minimum window size
    root.minsize(1000, 600)

    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    # Show welcome message
    print("🔬 Skin Disease Detection System - Inference Only")
    print("=" * 60)
    print("Application started successfully!")
    print("Load your pre-trained model to begin making predictions.")
    print("=" * 60)

    root.mainloop()


if __name__ == "__main__":
    main()
