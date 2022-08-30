import os

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from QDMpy import projectdir


class PyGPUfitNotInstalledDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("pyGPUfit needs to be installed")

        q_btn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(q_btn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel(
            "pyGPUfit needs to be installed. Please install it and try again. Try running:\n\n"
            f">>> pip install --no-index --find-links={os.path.join(projectdir, 'pyGpufit', 'pyGpufit-1.2.0-py2.py3-none-any.whl')} pyGpufit"
        )
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
