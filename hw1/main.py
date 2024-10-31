import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QGridLayout
from PyQt5.QtGui import QPixmap
import image_processing  # Import the module containing processing functions

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('OpenCV Homework')
        
        # Store paths for two images
        self.image_path1 = None
        self.image_path2 = None
        
        # Layouts
        main_layout = QVBoxLayout()
        
        # Image Loading Section
        load_layout = QHBoxLayout()
        self.load_btn1 = QPushButton("Load Image 1")
        self.load_btn1.clicked.connect(self.load_image1)
        load_layout.addWidget(self.load_btn1)
        
        self.load_btn2 = QPushButton("Load Image 2")
        self.load_btn2.clicked.connect(self.load_image2)
        load_layout.addWidget(self.load_btn2)
        
        main_layout.addLayout(load_layout)
        
        # Grid Layout for tasks
        grid_layout = QGridLayout()
        
        # Image Processing Buttons
        grid_layout.addWidget(QLabel("1. Image Processing"), 0, 0)
        grid_layout.addWidget(self.create_task_button("Color Separation", image_processing.color_separation), 1, 0)
        grid_layout.addWidget(self.create_task_button("Color Transformation", image_processing.color_transformation), 2, 0)
        grid_layout.addWidget(self.create_task_button("Color Extraction", image_processing.color_extraction), 3, 0)
        
        # Image Smoothing Buttons
        grid_layout.addWidget(QLabel("2. Image Smoothing"), 0, 1)
        grid_layout.addWidget(self.create_task_button("Gaussian Blur", image_processing.gaussian_blur), 1, 1)
        grid_layout.addWidget(self.create_task_button("Bilateral Filter", image_processing.bilateral_filter), 2, 1)
        grid_layout.addWidget(self.create_task_button("Median Filter", image_processing.median_filter), 3, 1)
        
        # Edge Detection Buttons
        grid_layout.addWidget(QLabel("3. Edge Detection"), 0, 2)
        grid_layout.addWidget(self.create_task_button("Sobel X", image_processing.sobel_x), 1, 2)
        grid_layout.addWidget(self.create_task_button("Sobel Y", image_processing.sobel_y), 2, 2)
        grid_layout.addWidget(self.create_task_button("Combination & Threshold", image_processing.combination_threshold), 3, 2)
        grid_layout.addWidget(self.create_task_button("Gradient Angle", image_processing.gradient_angle), 4, 2)
        
        # Transformation Section
        grid_layout.addWidget(QLabel("4. Transforms"), 0, 3)
        
        self.rotation_input = QLineEdit()
        self.rotation_input.setPlaceholderText("Degrees")
        grid_layout.addWidget(QLabel("Rotation:"), 1, 3)
        grid_layout.addWidget(self.rotation_input, 1, 4)
        
        self.scaling_input = QLineEdit()
        self.scaling_input.setPlaceholderText("Scale Factor")
        grid_layout.addWidget(QLabel("Scaling:"), 2, 3)
        grid_layout.addWidget(self.scaling_input, 2, 4)
        
        self.tx_input = QLineEdit()
        self.tx_input.setPlaceholderText("Tx Pixels")
        grid_layout.addWidget(QLabel("Tx:"), 3, 3)
        grid_layout.addWidget(self.tx_input, 3, 4)
        
        self.ty_input = QLineEdit()
        self.ty_input.setPlaceholderText("Ty Pixels")
        grid_layout.addWidget(QLabel("Ty:"), 4, 3)
        grid_layout.addWidget(self.ty_input, 4, 4)
        
        self.transform_btn = QPushButton("Apply Transform")
        self.transform_btn.clicked.connect(self.apply_transform)
        grid_layout.addWidget(self.transform_btn, 5, 3, 1, 2)
        
        main_layout.addLayout(grid_layout)
        self.setLayout(main_layout)
    
    def create_task_button(self, label, func):
        btn = QPushButton(label)
        btn.clicked.connect(lambda: func(self.image_path1))  # Use Image 1 by default
        return btn

    def load_image1(self):
        self.image_path1, _ = QFileDialog.getOpenFileName()
        
    def load_image2(self):
        self.image_path2, _ = QFileDialog.getOpenFileName()
    
    def apply_transform(self):
        if self.image_path1:
            try:
                angle = float(self.rotation_input.text())
                scale = float(self.scaling_input.text())
                tx = int(self.tx_input.text())
                ty = int(self.ty_input.text())
                image_processing.apply_transform(self.image_path1, angle, scale, tx, ty)
            except ValueError:
                print("Please enter valid numeric values.")

app = QApplication(sys.argv)
window = ImageProcessorApp()
window.show()
sys.exit(app.exec_())
