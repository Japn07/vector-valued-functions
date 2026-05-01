import PyInstaller.__main__
import os

print("Starting compilation of Math Viz Applet...")

PyInstaller.__main__.run([
    'main.py',                               # Main script
    '--name=Math_Viz_Multi_Applet',          # Name of the executable output
    '--windowed',                            # Don't open a console window when the app runs
    '--onefile',                             # Compresses everything into a single transportable file
    '--noconfirm',                           # Overwrite existing build folders
    '--hidden-import=numpy',
    '--hidden-import=matplotlib',
])

print("\nBuild Complete! You can find the executable inside the 'dist/Math_Viz_Multi_Applet' folder.")
