import tkinter as tk
from tkinter import ttk

from tabs.tnb_applet import TNBApp
from tabs.analysis_tools import AnalysisToolsTab

def main():
    root = tk.Tk()
    root.title("Vector Calculus Multi-Tool App")
    root.geometry("1400x850")
    
    # Enable slightly modernized styling for Notebook
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
        style.configure('TNotebook.Tab', font=('Segoe UI', 11, 'bold'), padding=[10, 5])
    except: pass
    
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)
    
    # 1. TNB Frame Applet Tab
    tnb_tab = TNBApp(notebook)
    notebook.add(tnb_tab, text='1. Interactive TNB Visualizer')
    
    # 2. Static Analysis Plotter Tab
    analysis_tab = AnalysisToolsTab(notebook)
    notebook.add(analysis_tab, text='2. Static Analysis Tools')
    

    # Launch main UI loop
    root.mainloop()

if __name__ == '__main__':
    main()
