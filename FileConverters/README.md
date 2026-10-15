
## Overview

This project converts .LIF files to .TIF, and provides a simple GUI to control its functions.

Has been tested on Python version 3.9 up to 3.12 and all work. In rare cases, it will crash at
tkinter.askopenfilename, but that seems to be an isolated problem that is solved by recreating the
environment.

To use this program, clone the repository, then install Python along with numpy, opencv, and Pillow:

    pip install numpy
    pip install opencv-python
    pip install Pillow

If the above commands don't work, try the following instead:

    python -m pip install numpy
    python -m pip install opencv-python
    python -m pip install Pillow

Then run the GUI:

    python -m LifConverterGUI

<div align="center"><img src="https://github.com/tomjhou/LIFconverter/blob/main/screenshot.png"></div>

There is also a console version, but I haven't updated it in a while, so it is not as functional as the GUI interface.

## CREATING EXE FILE FROM PYTHON:

If you want to create a standalone executable (for Windows or Mac), first install auto-py-to-exe:

    pip install auto-py-to-exe

You must install auto-py-to-exe in the SAME python environment that you will eventually run it in, e.g. don't
install under anaconda if you intend to run from Windows command prompt. Actually, anaconda is NOT
recommended as the .exe file will be about 4 times bigger (>200MB instead of 60MB).

This creates an .exe file in the Scripts folder of the python installation. For example, under anaconda, it might be here:

    C:/Users/TomJhou/anaconda3/Scripts/auto-py-to-exe.exe

(but I don't recommend that) or if installing under system Python, it will be somewhere like this:

    C:/Users/TomJhou/AppData/Local/Packages/PythonSoftwareFoundation_Python3.8_qbz5n2.../LocalCache/local-packages/Python38/Scripts

Add that folder to the Windows path if you want to run from command line. Or add a shortcut to taskbar and run auto-py-to-exe.exe from there.

When you run it, you have to select the "LifeConverterGUI.py" file, and it will put the final .exe file into a subfolder
called "output".


## NOTES ON READLIF

I've included a file "reader.py" that contains a modified version of Nick Negrtetti's "readlif" project.

The version on pypi.org is 0.6.5, which appears to be about 4 years old. There is a version on GitHub that seems newer,
but I haven't tried it:

    https://github.com/Arcadia-Science/readlif

Here are the changes I've made from the pypi.org version 0.6.5:

1. No longer crashes when reading newer lif files where the end of the file is zero instead of having one final magic number
2. No longer crashes if number of offsets is less than number of detected images (usually indicates truncated or corrupted file)
3. Fixed bug where magic number value error would close file and cause the wrong kind of error to be reported (truncation)

The python lif viewer is based on this java version:
https://github.com/ome/bioformats/blob/master/components/formats-gpl/src/loci/formats/in/LIFReader.java
