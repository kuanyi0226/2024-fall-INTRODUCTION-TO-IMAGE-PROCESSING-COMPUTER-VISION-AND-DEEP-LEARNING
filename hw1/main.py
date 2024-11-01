import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QGridLayout, QGroupBox
import image_processing  # Import the module containing processing functions

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hw1-MainWindow')
        
        # Store paths for two images
        self.image_path1 = None
        self.image_path2 = None    
        
        # Image Loading 
        self.load_btn1 = QPushButton("Load Image 1")
        self.load_btn1.clicked.connect(self.load_image1)
        self.load_label1 = QLabel("No image loaded")
        
        self.load_btn2 = QPushButton("Load Image 2")
        self.load_btn2.clicked.connect(self.load_image2)
        self.load_label2 = QLabel("No image loaded")
        
        #1. Image Processing Buttons
        problem1 = QGroupBox("1. Image Processing")
        layout1 = QVBoxLayout()
        #grid_layout.addWidget(QLabel("1. Image Processing"), 0, 0)
        layout1.addWidget(self.create_task_button("1.1 Color Separation", image_processing.color_separation))
        layout1.addWidget(self.create_task_button("1.2 Color Transformation", image_processing.color_transformation))
        layout1.addWidget(self.create_task_button("1.3 Color Extraction", image_processing.color_extraction)) 
        problem1.setLayout(layout1)

        #2. Image Smoothing Buttons
        problem2 = QGroupBox("2. Image Smoothing")
        layout2 = QVBoxLayout()
        #layout2.addWidget(QLabel("2. Image Smoothing"), 0, 1)
        layout2.addWidget(self.create_task_button("2.1 Gaussian Blur", image_processing.gaussian_blur))
        layout2.addWidget(self.create_task_button("2.2 Bilateral Filter", image_processing.bilateral_filter))
        layout2.addWidget(self.create_task_button("2.3 Median Filter", image_processing.median_filter))
        problem2.setLayout(layout2)

        #3. Edge Detection Buttons
        problem3 = QGroupBox("3. Edge Detection")
        layout3 = QVBoxLayout()
        #layout3.addWidget(QLabel("3. Edge Detection"), 0, 2)
        layout3.addWidget(self.create_task_button("3.1 Sobel X", image_processing.sobel_x))
        layout3.addWidget(self.create_task_button("3.2 Sobel Y", image_processing.sobel_y))
        layout3.addWidget(self.create_task_button("3.3 Combination and Threshold", image_processing.combination_threshold))
        layout3.addWidget(self.create_task_button("3.4 Gradient Angle", image_processing.gradient_angle))
        problem3.setLayout(layout3)

        #4. Transforms
        problem4 = QGroupBox("4. Transforms")
        layout4 = QVBoxLayout()
        transform_layout = QGridLayout()
        
        self.rotation_input = QLineEdit()
        transform_layout.addWidget(QLabel("Rotation:"), 1, 1)
        transform_layout.addWidget(self.rotation_input, 1, 2)
        transform_layout.addWidget(QLabel("deg"), 1, 3)
        
        self.scaling_input = QLineEdit()
        transform_layout.addWidget(QLabel("Scaling:"), 2, 1)
        transform_layout.addWidget(self.scaling_input, 2, 2)
        
        self.tx_input = QLineEdit()
        transform_layout.addWidget(QLabel("Tx:"), 3, 1)
        transform_layout.addWidget(self.tx_input, 3, 2)
        transform_layout.addWidget(QLabel("pixel"), 3, 3)
        
        self.ty_input = QLineEdit()
        transform_layout.addWidget(QLabel("Ty:"), 4, 1)
        transform_layout.addWidget(self.ty_input, 4, 2)
        transform_layout.addWidget(QLabel("pixel"), 4, 3)
        
        self.transform_btn = QPushButton("4. Transform")
        self.transform_btn.clicked.connect(self.apply_transform)
        transform_layout.addWidget(self.transform_btn, 5, 1, 1, 2)

        layout4.addLayout(transform_layout)
        problem4.setLayout(layout4)
        
        #main layout
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        mid_layout= QVBoxLayout()
        right_layout = QVBoxLayout()

        left_layout.addWidget(self.load_btn1)
        left_layout.addWidget(self.load_label1)
        left_layout.addWidget(self.load_btn2)
        left_layout.addWidget(self.load_label2)

        mid_layout.addWidget(problem1)
        mid_layout.addWidget(problem2)
        mid_layout.addWidget(problem3)

        right_layout.addWidget(problem4)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(mid_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)
    
    def create_task_button(self, label, func):
        btn = QPushButton(label)
        btn.clicked.connect(lambda: func(self.image_path1))  # Use Image 1 by default
        return btn

    def load_image1(self):
        self.image_path1, _ = QFileDialog.getOpenFileName()
        self.load_label1.setText(self.image_path1)
        
    def load_image2(self):
        self.image_path2, _ = QFileDialog.getOpenFileName()
        self.load_label2.setText(self.image_path2)
    
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
