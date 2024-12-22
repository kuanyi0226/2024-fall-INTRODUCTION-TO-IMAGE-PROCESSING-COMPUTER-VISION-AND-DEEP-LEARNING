import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
import train

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST and Cat-Dog Classifier")
        self.setGeometry(100, 100, 1000, 600)
        self.initUI()

    def initUI(self):
        # Main layout
        main_layout = QHBoxLayout()

        # Column 1: Load Image & Video
        col1 = QVBoxLayout()
        self.load_image_btn = QPushButton("Load Image")
        self.load_video_btn = QPushButton("Load Video")
        self.load_image_btn.clicked.connect(self.load_q1_image)
        col1.addWidget(self.load_image_btn)
        col1.addWidget(self.load_video_btn)

        # Column 2: Q1 Functions
        col2 = QVBoxLayout()
        self.q1_show_structure_btn = QPushButton("1.1 Show Structure")
        self.q1_show_acc_loss_btn = QPushButton("1.2 Show Acc and Loss")
        self.q1_predict_btn = QPushButton("1.3 Predict")
        self.q1_label = QLabel("Predict")

        self.q1_show_structure_btn.clicked.connect(train.load_and_show_vgg16)
        self.q1_show_acc_loss_btn.clicked.connect(lambda: train.show_accuracy_and_loss("mnist_vgg16_log.pth"))
        self.q1_predict_btn.clicked.connect(self.q1_predict)

        col2.addWidget(self.q1_show_structure_btn)
        col2.addWidget(self.q1_show_acc_loss_btn)
        col2.addWidget(self.q1_predict_btn)
        col2.addWidget(self.q1_label)

        # Column 3: Q2 Functions
        col3 = QVBoxLayout()
        self.q2_load_image_btn = QPushButton("Q2 Load Image")
        self.q2_1_btn = QPushButton("2.1 Show Images")
        self.q2_2_btn = QPushButton("2.2 Show Model Structure")
        self.q2_3_btn = QPushButton("2.3 Compare Models")
        self.q2_4_btn = QPushButton("2.4 Inference")
        self.q2_label = QLabel("Prediction:")

        self.q2_load_image_btn.clicked.connect(self.load_q2_image)
        self.q2_2_btn.clicked.connect(train.show_resnet50_architecture)
        self.q2_3_btn.clicked.connect(lambda: train.train_or_load_resnet50("model/Q2_resnet50_model.pth", "../Q2_Dataset/dataset/", epochs=10))
        self.q2_4_btn.clicked.connect(self.q2_inference)

        col3.addWidget(self.q2_load_image_btn)
        col3.addWidget(self.q2_1_btn)
        col3.addWidget(self.q2_2_btn)
        col3.addWidget(self.q2_3_btn)
        col3.addWidget(self.q2_4_btn)
        col3.addWidget(self.q2_label)

        # Column 4: Image Display
        col4 = QVBoxLayout()
        self.q1_image_label = QLabel()
        self.q1_image_label.setFixedSize(200, 200)
        self.q1_image_label.setStyleSheet("border: 1px solid black;")

        self.q2_image_label = QLabel()
        self.q2_image_label.setFixedSize(200, 200)
        self.q2_image_label.setStyleSheet("border: 1px solid black;")

        col4.addWidget(self.q1_image_label)
        col4.addWidget(self.q2_image_label)

        # Merge layouts
        main_layout.addLayout(col1)
        main_layout.addLayout(col2)
        main_layout.addLayout(col3)
        main_layout.addLayout(col4)

        # Set main widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_q1_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            pixmap = QPixmap(file_name)
            self.q1_image_label.setPixmap(pixmap.scaled(self.q1_image_label.size()))

    def q1_predict(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            result = train.predict_with_vgg16("best_mnist_vgg16.pth", file_name)
            self.q1_label.setText(f"Predict: {result}")

    def load_q2_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            pixmap = QPixmap(file_name)
            self.q2_image_label.setPixmap(pixmap.scaled(self.q2_image_label.size()))

    def q2_inference(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            result = train.predict_with_resnet50("model/Q2_resnet50_model.pth", file_name)
            self.q2_label.setText(f"Prediction: {result}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
