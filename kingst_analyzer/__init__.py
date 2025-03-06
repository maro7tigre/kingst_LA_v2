# In kingst_analyzer/init.py
import os
import sys
from pathlib import Path

from ._core import *  # Import all symbols from the C++ extension

version = "0.2.0"

# Get path to the DLLs relative to the package
package_dir = Path(file).parent
dll_dir = package_dir.parent / 'lib' / 'Win64'

# Add to DLL directories
if sys.version_info >= (3, 8):
    os.add_dll_directory(str(dll_dir))
else:
    # For older Python versions
    if sys.platform == 'win32':
        os.environ['PATH'] = str(dll_dir) + os.pathsep + os.environ['PATH']