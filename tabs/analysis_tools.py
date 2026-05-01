import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np

from math_physics import *

def plot_derivative_comparison(func, func_name, t_range=(0, 2*np.pi), is_3d=True):
    t_vals = np.linspace(t_range[0], t_range[1], 500)
    positions = np.array([func(t) for t in t_vals])
    velocities = np.array([numerical_derivative(func, t) for t in t_vals])
    accelerations = np.array([numerical_second_derivative(func, t) for t in t_vals])
    
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'Kinematic Analysis: {func_name}', fontsize=14)
    
    if is_3d:
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        
        ax1.plot(*positions.T, 'b-', linewidth=2)
        ax1.set_title('Position r(t)')
        ax2.plot(*velocities.T, 'g-', linewidth=2)
        ax2.set_title("Velocity r'(t)")
        ax3.plot(*accelerations.T, 'r-', linewidth=2)
        ax3.set_title("Acceleration r''(t)")
        
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    else:
        ax1 = fig.add_subplot(131); ax2 = fig.add_subplot(132); ax3 = fig.add_subplot(133)
        ax1.plot(*positions.T, 'b-', linewidth=2); ax1.set_title('Position r(t)')
        ax2.plot(*velocities.T, 'g-', linewidth=2); ax2.set_title("Velocity r'(t)")
        ax3.plot(*accelerations.T, 'r-', linewidth=2); ax3.set_title("Acceleration r''(t)")
        for ax in [ax1, ax2, ax3]:
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_component_functions(func, func_name, t_range=(0, 2*np.pi), is_3d=True):
    t_vals = np.linspace(t_range[0], t_range[1], 500)
    positions = np.array([func(t) for t in t_vals])
    
    n_components = 3 if is_3d else 2
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 3*n_components))
    if n_components == 1: axes = [axes]
    fig.suptitle(f'Component Functions: {func_name}', fontsize=14)
    
    labels = ['x(t)', 'y(t)', 'z(t)'] if is_3d else ['x(t)', 'y(t)']
    colors = ['blue', 'green', 'red']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(t_vals, positions[:, i], color=color, linewidth=2)
        ax.set_xlabel('t'); ax.set_ylabel(label); ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_curvature_along_curve(func, func_name, t_range=(0, 2*np.pi)):
    t_vals = np.linspace(t_range[0] + 0.1, t_range[1] - 0.1, 200)
    curvatures = [curvature(func, t) for t in t_vals]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_vals, curvatures, 'purple', linewidth=2)
    ax.set_xlabel('Parameter t'); ax.set_ylabel('Curvature κ')
    ax.set_title(f'Curvature Along Curve: {func_name}')
    ax.grid(True, alpha=0.3)
    ax.fill_between(t_vals, curvatures, alpha=0.3, color='purple')
    
    plt.tight_layout()
    plt.show()

def plot_arc_length_function(func, func_name, t_range=(0, 2*np.pi)):
    t_vals = np.linspace(t_range[0], t_range[1], 100)
    arc_lengths = [arc_length(func, t_range[0], t) for t in t_vals]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_vals, arc_lengths, 'teal', linewidth=2)
    ax.set_xlabel('Parameter t'); ax.set_ylabel('Arc Length s(t)')
    ax.set_title(f'Arc Length Function: {func_name}')
    ax.grid(True, alpha=0.3)
    ax.fill_between(t_vals, arc_lengths, alpha=0.3, color='teal')
    
    total_length = arc_lengths[-1]
    ax.annotate(f'Total Length: {total_length:.3f}',
                xy=(t_vals[-1], total_length),
                xytext=(t_vals[-1]*0.7, total_length*0.8),
                fontsize=12,
                arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    plt.show()

class AnalysisToolsTab(ttk.Frame):
    SAFE_NAMESPACE = {
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
        'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
        'exp': np.exp, 'log': np.log, 'ln': np.log, 'log2': np.log2, 'log10': np.log10,
        'sqrt': np.sqrt, 'abs': np.abs,
        'pi': np.pi, 'e': np.e,
        'np': np,
    }

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.preset_funcs = {
            'Helix': helix,
            'Circle (2D)': circle_2d,
            'Trefoil Knot': trefoil_knot,
            'Torus Knot': torus_knot,
            'Lissajous (2D)': lissajous_2d,
            'Cycloid (2D)': cycloid_2d,
            'Spiral (2D)': spiral_2d,
            'Parabola (3D)': parabola_3d
        }
        
        lbl = ttk.Label(self, text='Static Analysis Tools', font=('Segoe UI', 14, 'bold'))
        lbl.pack(pady=10)
        
        frame = ttk.Frame(self)
        frame.pack(pady=5)
        
        ttk.Label(frame, text='Target Curve:').pack(side=tk.LEFT, padx=5)
        self.curve_var = tk.StringVar(value='Helix')
        cb = ttk.Combobox(frame, textvariable=self.curve_var, values=list(self.preset_funcs.keys()) + ['Custom'], state='readonly')
        cb.pack(side=tk.LEFT, padx=5)
        
        # Custom expression area
        custom_frame = ttk.LabelFrame(self, text="Custom r(t) Input (Active when Target Curve = 'Custom')")
        custom_frame.pack(pady=10, padx=20, fill=tk.X)
        
        def make_entry(row, label_text, default_val):
            ttk.Label(custom_frame, text=label_text).grid(row=row, column=0, padx=5, pady=2, sticky='e')
            ent = ttk.Entry(custom_frame, width=30)
            ent.grid(row=row, column=1, padx=5, pady=2, sticky='w')
            ent.insert(0, default_val)
            return ent
            
        self.entry_x = make_entry(0, "x(t) =", "t*cos(t)")
        self.entry_y = make_entry(1, "y(t) =", "t*sin(t)")
        self.entry_z = make_entry(2, "z(t) = (leave blank for 2D)", "t")
        
        tr_frame = ttk.Frame(custom_frame)
        tr_frame.grid(row=3, column=0, columnspan=2, pady=5)
        ttk.Label(tr_frame, text="t range: [").pack(side=tk.LEFT)
        self.entry_tmin = ttk.Entry(tr_frame, width=8)
        self.entry_tmin.insert(0, "0")
        self.entry_tmin.pack(side=tk.LEFT)
        ttk.Label(tr_frame, text=", ").pack(side=tk.LEFT)
        self.entry_tmax = ttk.Entry(tr_frame, width=8)
        self.entry_tmax.insert(0, "4*pi")
        self.entry_tmax.pack(side=tk.LEFT)
        ttk.Label(tr_frame, text="]").pack(side=tk.LEFT)
        
        btns_frame = ttk.Frame(self)
        btns_frame.pack(pady=10, fill=tk.X, padx=100)
        
        ttk.Button(btns_frame, text='Derivative Comparison', command=self._do_derivatives).pack(pady=2, fill=tk.X)
        ttk.Button(btns_frame, text='Component Functions', command=self._do_components).pack(pady=2, fill=tk.X)
        ttk.Button(btns_frame, text='Curvature Plot', command=self._do_curvature).pack(pady=2, fill=tk.X)
        ttk.Button(btns_frame, text='Arc Length Plot', command=self._do_arclength).pack(pady=2, fill=tk.X)

    def _parse_expr(self, expr_str):
        code = compile(expr_str, '<user_input>', 'eval')
        def f(t):
            ns = {**self.SAFE_NAMESPACE, 't': t}
            return eval(code, ns)
        return f

    def _get_active(self):
        name = self.curve_var.get()
        if name == 'Custom':
            x_str = self.entry_x.get().strip()
            y_str = self.entry_y.get().strip()
            z_str = self.entry_z.get().strip()
            
            fx = self._parse_expr(x_str)
            fy = self._parse_expr(y_str)
            
            if z_str:
                fz = self._parse_expr(z_str)
                func = lambda t: np.array([fx(t), fy(t), fz(t)], dtype=float)
                is_3d = True
            else:
                func = lambda t: np.array([fx(t), fy(t)], dtype=float)
                is_3d = False
                
            tmin = float(eval(self.entry_tmin.get(), {**self.SAFE_NAMESPACE}))
            tmax = float(eval(self.entry_tmax.get(), {**self.SAFE_NAMESPACE}))
            return func, 'Custom', (tmin, tmax), is_3d
            
        else:
            func = self.preset_funcs[name]
            is_3d = '(2D)' not in name
            
            ranges = {
                'Helix': (0, 4*np.pi),
                'Circle (2D)': (0, 2*np.pi),
                'Trefoil Knot': (0, 2*np.pi),
                'Torus Knot': (0, 2*np.pi),
                'Lissajous (2D)': (0, 2*np.pi),
                'Cycloid (2D)': (0, 4*np.pi),
                'Spiral (2D)': (0, 6*np.pi),
                'Parabola (3D)': (-2, 2)
            }
            trange = ranges.get(name, (0, 2*np.pi))
            return func, name, trange, is_3d

    def _do_derivatives(self):
        f, n, tr, i3 = self._get_active()
        plot_derivative_comparison(f, n, tr, i3)

    def _do_components(self):
        f, n, tr, i3 = self._get_active()
        plot_component_functions(f, n, tr, i3)

    def _do_curvature(self):
        f, n, tr, i3 = self._get_active()
        plot_curvature_along_curve(f, n, tr)

    def _do_arclength(self):
        f, n, tr, i3 = self._get_active()
        plot_arc_length_function(f, n, tr)
