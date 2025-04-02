import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox


def run_files_in_sequence():
    try:
        # 按顺序运行三个 Python 文件
        file_names = ['getdata.py', 'data_preparation.py', 'train_model.py']
        for file in file_names:
            print(f"正在运行 {file}...")
            subprocess.run([sys.executable, file], check=True)
            print(f"{file} 运行完成。")
        # 所有文件运行成功后显示信息采集成功的提示框
        QMessageBox.information(None, "提示", "信息采集成功")
    except subprocess.CalledProcessError as e:
        print(f"运行 {e.cmd} 时出错: {e}")
        QMessageBox.critical(None, "错误", f"运行 {e.cmd} 时出错: {e}")


def run_MainUI():
    try:
        # 调用 MainUI.py 文件
        subprocess.run([sys.executable, 'MainUI.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"运行 MainUI.py 时出错: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 创建主窗口
    window = QWidget()
    window.setWindowTitle("功能选择窗口")
    window.resize(400, 300)

    # 创建布局
    layout = QVBoxLayout()

    # 创建“采集信息”按钮
    collect_info_button = QPushButton("采集信息")
    collect_info_button.clicked.connect(run_files_in_sequence)
    layout.addWidget(collect_info_button)

    # 创建“人物识别”按钮
    person_recognition_button = QPushButton("人物识别")
    person_recognition_button.clicked.connect(run_MainUI)
    layout.addWidget(person_recognition_button)

    # 设置布局
    window.setLayout(layout)

    # 显示窗口
    window.show()

    # 运行应用程序
    sys.exit(app.exec_())