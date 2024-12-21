# main.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
import train

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST and Cat-Dog Classifier")
        self.setGeometry(100, 100, 800, 600)

        # Layout
        layout = QVBoxLayout()

        # Buttons for Task 1
        btn1_1 = QPushButton("1.1 Show Structure")
        btn1_1.clicked.connect(train.load_and_show_vgg16)
        layout.addWidget(btn1_1)

        btn1_2 = QPushButton("1.2 Show Acc and Loss")
        btn1_2.clicked.connect(train.show_accuracy_and_loss)
        layout.addWidget(btn1_2)

        btn1_3 = QPushButton("1.3 Predict")
        btn1_3.clicked.connect(lambda: train.predict_with_vgg16(model_path='best_mnist_model.h5', sample_image_path='sample_image.png'))
        layout.addWidget(btn1_3)

        # Buttons for Task 2
        btn2_1 = QPushButton("2.1 Load Image")
        btn2_1.clicked.connect(lambda: train.load_and_resize_dataset('cat_dog_dataset/'))
        layout.addWidget(btn2_1)

        btn2_2 = QPushButton("2.2 Show Model Structure")
        btn2_2.clicked.connect(train.load_and_show_resnet50)
        layout.addWidget(btn2_2)

        btn2_3 = QPushButton("2.3 Show Comparison")
        btn2_3.clicked.connect(lambda: train.train_and_compare_resnet50('cat_dog_dataset/'))
        layout.addWidget(btn2_3)

        btn2_4 = QPushButton("2.4 Inference")
        btn2_4.clicked.connect(lambda: train.predict_with_resnet50(model_path='resnet50_model.h5', sample_image_path='cat_or_dog.png'))
        layout.addWidget(btn2_4)

        # Central Widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
