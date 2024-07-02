import os 

with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as file:
    __version__ = file.read().strip()