import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from math_physics import *


class TNBApp(ttk.Frame):
    """Interactive Tkinter applet for TNB frame visualization."""

    # Safe namespace for evaluating user expressions
    SAFE_NAMESPACE = {
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
        'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
        'exp': np.exp, 'log': np.log, 'ln': np.log, 'log2': np.log2, 'log10': np.log10,
        'sqrt': np.sqrt, 'abs': np.abs,
        'pi': np.pi, 'e': np.e,
        'np': np,
    }

    PRESETS = {
        'Helix': ('cos(t)', 'sin(t)', 't / 5', '0', '4*pi'),
        'Circle (2D)': ('cos(t)', 'sin(t)', '', '0', '2*pi'),
        'Trefoil Knot': ('sin(t) + 2*sin(2*t)', 'cos(t) - 2*cos(2*t)', '-sin(3*t)', '0', '2*pi'),
        'Torus Knot (2,3)': ('(cos(3*t)+2)*cos(2*t)', '(cos(3*t)+2)*sin(2*t)', '-sin(3*t)', '0', '2*pi'),
        'Lissajous (2D)': ('sin(3*t + pi/2)', 'sin(2*t)', '', '0', '2*pi'),
        'Cycloid (2D)': ('t - sin(t)', '1 - cos(t)', '', '0', '4*pi'),
        'Spiral (2D)': ('0.1*t*cos(t)', '0.1*t*sin(t)', '', '0', '6*pi'),
        'Viviani Curve': ('1 + cos(t)', 'sin(t)', '2*sin(t/2)', '0', '2*pi'),
        'Parabolic 3D': ('t', 't**2', 't**3', '-2', '2'),
    }

    PATH_COLORS = ['#e06c75', '#61afef', '#98c379', '#c678dd', '#e5c07b',
                   '#56b6c2', '#be5046', '#d19a66', '#7ec8e3', '#c3e88d']

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self._current_func = None
        self._is_3d = True
        self._hover_annotation = None
        self._quiver_artists = []

        # Multi-path state: list of dicts {x, y, z, tmin, tmax, name}
        self._paths = []
        self._selected_idx = 0

        self._build_styles()
        self._build_ui()
        self._add_path_from_preset('Helix')
        self._do_plot()

    # ── Styles ───────────────────────────────────────────────────────────

    def _build_styles(self):
        self.BG = '#1e1e2e'
        self.BG2 = '#282840'
        self.FG = '#cdd6f4'
        self.ACCENT = '#89b4fa'
        self.GREEN = '#a6e3a1'
        self.RED = '#f38ba8'
        self.YELLOW = '#f9e2af'
        self.MAUVE = '#cba6f7'
        self.TEAL = '#94e2d5'
        self.PEACH = '#fab387'

        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except: pass

        style.configure('App.TFrame', background=self.BG)
        style.configure('Card.TFrame', background=self.BG2)
        style.configure('App.TLabel', background=self.BG, foreground=self.FG,
                        font=('Segoe UI', 10))
        style.configure('Card.TLabel', background=self.BG2, foreground=self.FG,
                        font=('Segoe UI', 10))
        style.configure('Header.TLabel', background=self.BG, foreground=self.ACCENT,
                        font=('Segoe UI', 13, 'bold'))
        style.configure('Info.TLabel', background=self.BG2, foreground=self.FG,
                        font=('Consolas', 9))
        style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'))
        style.configure('App.TCheckbutton', background=self.BG, foreground=self.FG,
                        font=('Segoe UI', 10))
        style.map('App.TCheckbutton',
                  background=[('active', self.BG)],
                  foreground=[('active', self.ACCENT)])

    # ── UI Construction ──────────────────────────────────────────────────

    def _build_ui(self):
        # Main horizontal panes
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ── Left panel (scrollable controls) ──
        left_outer = ttk.Frame(main, style='App.TFrame', width=340)
        main.add(left_outer, weight=0)

        # Scrollable setup: Canvas + Scrollbar + inner Frame
        self._left_canvas = tk.Canvas(left_outer, bg=self.BG, highlightthickness=0,
                                       width=340)
        self._left_scrollbar = ttk.Scrollbar(left_outer, orient=tk.VERTICAL,
                                              command=self._left_canvas.yview)
        self._left_scroll_frame = ttk.Frame(self._left_canvas, style='App.TFrame')

        self._left_scroll_frame.bind(
            '<Configure>',
            lambda e: self._left_canvas.configure(
                scrollregion=self._left_canvas.bbox('all')))

        self._left_canvas_window = self._left_canvas.create_window(
            (0, 0), window=self._left_scroll_frame, anchor='nw')

        # Make inner frame match canvas width
        self._left_canvas.bind('<Configure>',
            lambda e: self._left_canvas.itemconfig(
                self._left_canvas_window, width=e.width))

        self._left_canvas.configure(yscrollcommand=self._left_scrollbar.set)

        self._left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mousewheel scrolling
        def _on_mousewheel(event):
            self._left_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        self._left_canvas.bind_all('<MouseWheel>', _on_mousewheel)

        # ── Right panel (plot) ──
        right = ttk.Frame(main, style='App.TFrame')
        main.add(right, weight=1)

        self._build_controls(self._left_scroll_frame)
        self._build_canvas(right)

    def _build_controls(self, parent):
        # Title
        ttk.Label(parent, text='TNB Frame Visualizer', style='Header.TLabel').pack(
            pady=(8, 12), anchor='w', padx=8)

        # ── Paths list ──
        paths_card = ttk.Frame(parent, style='Card.TFrame', padding=8)
        paths_card.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Label(paths_card, text='Paths', style='Card.TLabel',
                  font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(0, 4))

        self._path_listbox = tk.Listbox(paths_card, height=4, bg='#313244',
                                         fg=self.FG, selectbackground=self.ACCENT,
                                         selectforeground='#1e1e2e', relief='flat',
                                         font=('Consolas', 9), exportselection=False)
        self._path_listbox.pack(fill=tk.X, pady=(0, 4))
        self._path_listbox.bind('<<ListboxSelect>>', self._on_path_select)

        btn_row = ttk.Frame(paths_card, style='Card.TFrame')
        btn_row.pack(fill=tk.X)
        tk.Button(btn_row, text='+ Add', bg=self.GREEN, fg='#1e1e2e',
                  font=('Segoe UI', 9, 'bold'), bd=0, padx=8, pady=2,
                  command=self._add_current_as_path).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_row, text='− Remove', bg=self.RED, fg='#1e1e2e',
                  font=('Segoe UI', 9, 'bold'), bd=0, padx=8, pady=2,
                  command=self._remove_selected_path).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_row, text='Update', bg=self.YELLOW, fg='#1e1e2e',
                  font=('Segoe UI', 9, 'bold'), bd=0, padx=8, pady=2,
                  command=self._update_selected_path).pack(side=tk.LEFT)

        # ── Preset selector (adds as new path) ──
        pf = ttk.Frame(parent, style='App.TFrame')
        pf.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(pf, text='Add Preset:', style='App.TLabel').pack(side=tk.LEFT)
        self._preset_var = tk.StringVar(value='')
        pcb = ttk.Combobox(pf, textvariable=self._preset_var,
                          values=list(self.PRESETS.keys()),
                          state='readonly', width=18)
        pcb.pack(side=tk.LEFT, padx=(6, 0))
        pcb.bind('<<ComboboxSelected>>', lambda e: self._add_path_from_preset(self._preset_var.get()))

        # ── r(t) input fields (edit selected path) ──
        card = ttk.Frame(parent, style='Card.TFrame', padding=10)
        card.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(card, text='r(t) Components', style='Card.TLabel',
                  font=('Segoe UI', 11, 'bold')).grid(row=0, column=0, columnspan=2,
                                                       sticky='w', pady=(0, 6))

        self._entry_x = self._make_entry(card, 'x(t) =', 1)
        self._entry_y = self._make_entry(card, 'y(t) =', 2)
        self._entry_z = self._make_entry(card, 'z(t) =', 3)

        ttk.Label(card, text='(leave z blank for 2D)', style='Card.TLabel',
                  font=('Segoe UI', 8)).grid(row=4, column=1, sticky='w', pady=(0, 4))

        # ── t range ──
        rf = ttk.Frame(card, style='Card.TFrame')
        rf.grid(row=5, column=0, columnspan=2, sticky='ew', pady=(4, 0))
        ttk.Label(rf, text='t ∈ [', style='Card.TLabel').pack(side=tk.LEFT)
        self._entry_tmin = tk.Entry(rf, width=8, bg='#313244', fg=self.FG,
                                    insertbackground=self.FG, relief='flat', font=('Consolas', 10))
        self._entry_tmin.pack(side=tk.LEFT, padx=2)
        self._entry_tmin.insert(0, '0')
        ttk.Label(rf, text=',', style='Card.TLabel').pack(side=tk.LEFT)
        self._entry_tmax = tk.Entry(rf, width=8, bg='#313244', fg=self.FG,
                                    insertbackground=self.FG, relief='flat', font=('Consolas', 10))
        self._entry_tmax.pack(side=tk.LEFT, padx=2)
        self._entry_tmax.insert(0, '4*pi')
        ttk.Label(rf, text=']', style='Card.TLabel').pack(side=tk.LEFT)

        # ── Plot button ──
        btn = tk.Button(parent, text='▶  Plot', bg=self.ACCENT, fg='#1e1e2e',
                        activebackground='#b4d0fb', activeforeground='#1e1e2e',
                        font=('Segoe UI', 11, 'bold'), bd=0, padx=16, pady=6,
                        cursor='hand2', command=self._do_plot)
        btn.pack(fill=tk.X, padx=8, pady=8)
        for entry in (self._entry_x, self._entry_y, self._entry_z,
                      self._entry_tmin, self._entry_tmax):
            entry.bind('<Return>', lambda e: self._do_plot())

        # ── t₀ slider ──
        sf = ttk.Frame(parent, style='App.TFrame')
        sf.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(sf, text='t₀  (selected path)', style='App.TLabel',
                  font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        self._t0_var = tk.DoubleVar(value=1.0)
        self._slider = tk.Scale(sf, from_=0, to=4*np.pi, resolution=0.01,
                                orient=tk.HORIZONTAL, variable=self._t0_var,
                                bg=self.BG, fg=self.FG, troughcolor=self.BG2,
                                highlightthickness=0, bd=0,
                                activebackground=self.ACCENT,
                                font=('Consolas', 9),
                                command=self._on_slider)
        self._slider.pack(fill=tk.X)

        # ── Vector checkboxes ──
        vf = ttk.Frame(parent, style='App.TFrame')
        vf.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(vf, text='Show Vectors', style='App.TLabel',
                  font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(0, 4))

        self._show_T = tk.BooleanVar(value=True)
        self._show_N = tk.BooleanVar(value=True)
        self._show_B = tk.BooleanVar(value=True)
        self._show_v = tk.BooleanVar(value=False)
        self._show_a = tk.BooleanVar(value=False)

        checks = [
            ('T  (Unit Tangent)', self._show_T, '#3b82f6'),
            ('N  (Principal Normal)', self._show_N, '#22c55e'),
            ('B  (Binormal)', self._show_B, '#f97316'),
            ("v  (Velocity r')", self._show_v, '#a855f7'),
            ("a  (Acceleration r'')", self._show_a, '#06b6d4'),
        ]
        for label, var, color in checks:
            f = ttk.Frame(vf, style='App.TFrame')
            f.pack(fill=tk.X, pady=1)
            cb = ttk.Checkbutton(f, text=label, variable=var,
                                 style='App.TCheckbutton',
                                 command=self._update_plot)
            cb.pack(side=tk.LEFT)
            indicator = tk.Canvas(f, width=14, height=14, bg=self.BG,
                                 highlightthickness=0)
            indicator.create_oval(2, 2, 12, 12, fill=color, outline='')
            indicator.pack(side=tk.RIGHT, padx=4)

        # ── Vector scale ──
        scf = ttk.Frame(parent, style='App.TFrame')
        scf.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(scf, text='Vector Scale', style='App.TLabel').pack(anchor='w')
        self._scale_var = tk.DoubleVar(value=0.5)
        self._scale_slider = tk.Scale(scf, from_=0.1, to=3.0, resolution=0.05,
                                      orient=tk.HORIZONTAL, variable=self._scale_var,
                                      bg=self.BG, fg=self.FG, troughcolor=self.BG2,
                                      highlightthickness=0, bd=0,
                                      activebackground=self.ACCENT,
                                      font=('Consolas', 9),
                                      command=self._on_slider)
        self._scale_slider.pack(fill=tk.X)

        # ── Prominent κ / τ / L display ──
        kt_card = ttk.Frame(parent, style='Card.TFrame', padding=10)
        kt_card.pack(fill=tk.X, padx=8, pady=(10, 4))

        kt_row1 = ttk.Frame(kt_card, style='Card.TFrame')
        kt_row1.pack(fill=tk.X, pady=2)
        ttk.Label(kt_row1, text='κ  =', style='Card.TLabel',
                  font=('Segoe UI', 13, 'bold')).pack(side=tk.LEFT)
        self._kappa_label = ttk.Label(kt_row1, text='—', style='Card.TLabel',
                                      font=('Consolas', 14, 'bold'),
                                      foreground='#f9e2af')
        self._kappa_label.pack(side=tk.LEFT, padx=(6, 0))

        kt_row2 = ttk.Frame(kt_card, style='Card.TFrame')
        kt_row2.pack(fill=tk.X, pady=2)
        ttk.Label(kt_row2, text='τ  =', style='Card.TLabel',
                  font=('Segoe UI', 13, 'bold')).pack(side=tk.LEFT)
        self._tau_label = ttk.Label(kt_row2, text='—', style='Card.TLabel',
                                    font=('Consolas', 14, 'bold'),
                                    foreground='#94e2d5')
        self._tau_label.pack(side=tk.LEFT, padx=(6, 0))

        kt_row3 = ttk.Frame(kt_card, style='Card.TFrame')
        kt_row3.pack(fill=tk.X, pady=2)
        ttk.Label(kt_row3, text='L  =', style='Card.TLabel',
                  font=('Segoe UI', 13, 'bold')).pack(side=tk.LEFT)
        self._arclen_label = ttk.Label(kt_row3, text='—', style='Card.TLabel',
                                       font=('Consolas', 14, 'bold'),
                                       foreground='#fab387')
        self._arclen_label.pack(side=tk.LEFT, padx=(6, 0))

        # ── Info panel ──
        info_card = ttk.Frame(parent, style='Card.TFrame', padding=8)
        info_card.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 8))
        ttk.Label(info_card, text='Calculus Data (selected path)',
                  style='Card.TLabel',
                  font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(0, 4))
        self._info_label = ttk.Label(info_card, text='', style='Info.TLabel',
                                     justify=tk.LEFT, wraplength=300)
        self._info_label.pack(fill=tk.BOTH, expand=True, anchor='nw')

    def _make_entry(self, parent, label, row):
        ttk.Label(parent, text=label, style='Card.TLabel',
                  font=('Consolas', 11)).grid(row=row, column=0, sticky='w', padx=(0, 4), pady=2)
        entry = tk.Entry(parent, width=28, bg='#313244', fg=self.FG,
                         insertbackground=self.FG, relief='flat',
                         font=('Consolas', 11))
        entry.grid(row=row, column=1, sticky='ew', pady=2)
        return entry

    def _build_canvas(self, parent):
        self._fig = Figure(figsize=(7, 6), facecolor=self.BG)
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas.mpl_connect('motion_notify_event', self._on_hover)

    # ── Path management ───────────────────────────────────────────────────

    def _make_path_dict(self, x, y, z, tmin, tmax, name=None):
        if name is None:
            name = f'⟨{x}, {y}, {z}⟩' if z else f'⟨{x}, {y}⟩'
        return {'x': x, 'y': y, 'z': z, 'tmin': tmin, 'tmax': tmax, 'name': name}

    def _refresh_listbox(self):
        self._path_listbox.delete(0, tk.END)
        for i, p in enumerate(self._paths):
            color = self.PATH_COLORS[i % len(self.PATH_COLORS)]
            self._path_listbox.insert(tk.END, f'[{i+1}] {p["name"]}')
            self._path_listbox.itemconfig(i, fg=color)
        if self._paths:
            self._selected_idx = min(self._selected_idx, len(self._paths) - 1)
            self._path_listbox.selection_set(self._selected_idx)

    def _load_entries_from_path(self, p):
        for entry, key in [(self._entry_x, 'x'), (self._entry_y, 'y'),
                           (self._entry_z, 'z'), (self._entry_tmin, 'tmin'),
                           (self._entry_tmax, 'tmax')]:
            entry.delete(0, tk.END)
            entry.insert(0, p[key])

    def _add_path_from_preset(self, name):
        x, y, z, tmin, tmax = self.PRESETS[name]
        p = self._make_path_dict(x, y, z, tmin, tmax, name)
        self._paths.append(p)
        self._selected_idx = len(self._paths) - 1
        self._refresh_listbox()
        self._load_entries_from_path(p)
        self._do_plot()

    def _add_current_as_path(self):
        x, y = self._entry_x.get().strip(), self._entry_y.get().strip()
        z = self._entry_z.get().strip()
        tmin, tmax = self._entry_tmin.get().strip(), self._entry_tmax.get().strip()
        if not x or not y:
            return
        self._paths.append(self._make_path_dict(x, y, z, tmin, tmax))
        self._selected_idx = len(self._paths) - 1
        self._refresh_listbox()
        self._do_plot()

    def _remove_selected_path(self):
        if not self._paths:
            return
        del self._paths[self._selected_idx]
        if self._paths:
            self._selected_idx = min(self._selected_idx, len(self._paths) - 1)
            self._refresh_listbox()
            self._load_entries_from_path(self._paths[self._selected_idx])
        else:
            self._selected_idx = 0
            self._refresh_listbox()
        self._do_plot()

    def _update_selected_path(self):
        if not self._paths:
            return
        x, y = self._entry_x.get().strip(), self._entry_y.get().strip()
        z = self._entry_z.get().strip()
        tmin, tmax = self._entry_tmin.get().strip(), self._entry_tmax.get().strip()
        if not x or not y:
            return
        self._paths[self._selected_idx] = self._make_path_dict(x, y, z, tmin, tmax)
        self._refresh_listbox()
        self._do_plot()

    def _on_path_select(self, event):
        sel = self._path_listbox.curselection()
        if not sel:
            return
        self._selected_idx = sel[0]
        self._load_entries_from_path(self._paths[self._selected_idx])
        tmin, tmax = self._parse_t_range_for(self._paths[self._selected_idx])
        self._slider.configure(from_=tmin, to=tmax)
        t0 = self._t0_var.get()
        if t0 < tmin or t0 > tmax:
            self._t0_var.set((tmin + tmax) / 2)
        self._update_plot()

    # ── Expression parsing ───────────────────────────────────────────────

    def _parse_expr(self, expr_str):
        code = compile(expr_str, '<user_input>', 'eval')
        def f(t):
            ns = {**self.SAFE_NAMESPACE, 't': t}
            return eval(code, ns)
        return f

    def _build_func_from_path(self, p):
        fx, fy = self._parse_expr(p['x']), self._parse_expr(p['y'])
        if p['z']:
            fz = self._parse_expr(p['z'])
            def func(t):
                return np.array([fx(t), fy(t), fz(t)], dtype=float)
            return func, True
        else:
            def func(t):
                return np.array([fx(t), fy(t)], dtype=float)
            return func, False

    def _parse_t_range_for(self, p):
        ns = {**self.SAFE_NAMESPACE}
        return (float(eval(p['tmin'], ns)),
                float(eval(p['tmax'], ns)))

    def _parse_t_range(self):
        ns = {**self.SAFE_NAMESPACE}
        return (float(eval(self._entry_tmin.get().strip(), ns)),
                float(eval(self._entry_tmax.get().strip(), ns)))

    # ── Plotting ─────────────────────────────────────────────────────────

    def _do_plot(self, *_):
        if not self._paths:
            self._fig.clear(); self._canvas.draw()
            return
        try:
            self._is_3d = any(p['z'] for p in self._paths)
            sel_p = self._paths[self._selected_idx]
            tmin, tmax = self._parse_t_range_for(sel_p)
            self._slider.configure(from_=tmin, to=tmax)
            t0 = self._t0_var.get()
            if t0 < tmin or t0 > tmax:
                self._t0_var.set((tmin + tmax) / 2)
            self._current_func, _ = self._build_func_from_path(sel_p)
            self._update_plot()
        except Exception as e:
            self._info_label.configure(text=f'Error: {e}')

    def _on_slider(self, *_):
        if self._paths:
            self._update_plot()

    def _update_plot(self):
        if not self._paths:
            return
        try:
            self._fig.clear()
            scale = self._scale_var.get()
            sel_p = self._paths[self._selected_idx]
            sel_tmin, sel_tmax = self._parse_t_range_for(sel_p)
            t0 = self._t0_var.get()
            sel_func, sel_is_3d = self._build_func_from_path(sel_p)
            self._current_func = sel_func

            if self._is_3d:
                ax = self._fig.add_subplot(111, projection='3d', facecolor=self.BG)
            else:
                ax = self._fig.add_subplot(111, facecolor=self.BG)

            all_pts = []
            for i, p in enumerate(self._paths):
                color = self.PATH_COLORS[i % len(self.PATH_COLORS)]
                try:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        fi, _ = self._build_func_from_path(p)
                        ti, tx = self._parse_t_range_for(p)
                        pts = np.array([fi(t) for t in np.linspace(ti, tx, 500)])
                    
                    # Filter out NaN or Inf points
                    valid_mask = np.all(np.isfinite(pts), axis=1)
                    pts = pts[valid_mask]
                    if len(pts) == 0:
                        continue

                    if self._is_3d and pts.shape[1] == 2:
                        pts = np.column_stack([pts, np.zeros(len(pts))])
                    all_pts.append(pts)

                    lw = 2.5 if i == self._selected_idx else 1.5
                    al = 1.0 if i == self._selected_idx else 0.5
                    if self._is_3d:
                        ax.plot(pts[:,0], pts[:,1], pts[:,2], color=color,
                                linewidth=lw, alpha=al, label=p['name'])
                    else:
                        ax.plot(pts[:,0], pts[:,1], color=color,
                                linewidth=lw, alpha=al, label=p['name'])
                except Exception:
                    continue

            if self._is_3d and all_pts:
                c = np.vstack(all_pts)
                # Ensure we have valid points to bound
                if len(c) > 0 and np.any(np.isfinite(c)):
                    pad = np.max(np.abs(c)) * 1.3
                    ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad); ax.set_zlim(-pad, pad)
                ax.set_xlabel('X', color=self.FG); ax.set_ylabel('Y', color=self.FG)
                ax.set_zlabel('Z', color=self.FG); ax.tick_params(colors=self.FG)
                ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
            elif not self._is_3d:
                ax.autoscale(); ax.set_aspect('equal')
                ax.set_xlabel('X', color=self.FG); ax.set_ylabel('Y', color=self.FG)
                ax.tick_params(colors=self.FG); ax.grid(True, alpha=0.15, color=self.FG)

            with np.errstate(divide='ignore', invalid='ignore'):
                pos = sel_func(t0)
                pos_valid = np.all(np.isfinite(pos))

                if pos_valid:
                    if self._is_3d and len(pos) == 2:
                        pos = np.append(pos, 0.0)
                    if self._is_3d:
                        ax.scatter(*pos[:3], color=self.RED, s=80, zorder=5, depthshade=False)
                    else:
                        ax.scatter(*pos[:2], color=self.RED, s=80, zorder=5)

                self._quiver_artists = []
                if pos_valid:
                    vec_cfg = []
                    if self._show_T.get():
                        T_r = tangent_vector(sel_func, t0, normalize=True)
                        vec_cfg.append((T_r * scale, '#3b82f6', 'T (Unit Tangent)', T_r))
                    if self._show_N.get():
                        N_r = normal_vector(sel_func, t0)
                        vec_cfg.append((N_r * scale, '#22c55e', 'N (Principal Normal)', N_r))
                    if self._show_B.get() and sel_is_3d:
                        B_r = binormal_vector(sel_func, t0)
                        vec_cfg.append((B_r * scale, '#f97316', 'B (Binormal)', B_r))
                    if self._show_v.get():
                        v_r = numerical_derivative(sel_func, t0)
                        vec_cfg.append((v_r * scale * 0.5, '#a855f7', "v (Velocity r')", v_r))
                    if self._show_a.get():
                        a_r = numerical_second_derivative(sel_func, t0)
                        vec_cfg.append((a_r * scale * 0.3, '#06b6d4', "a (Acceleration r'')", a_r))

                    for vec, clr, lbl, raw in vec_cfg:
                        if not np.all(np.isfinite(vec)): continue
                        dv, dp = vec, pos
                        if self._is_3d and len(vec) == 2:
                            dv = np.append(vec, 0.0)
                        if self._is_3d:
                            q = ax.quiver(*dp[:3], *dv[:3], color=clr, arrow_length_ratio=0.18,
                                          linewidth=2.2, label=lbl.split(' (')[0])
                        else:
                            q = ax.quiver(*dp[:2], *dv[:2], color=clr, angles='xy',
                                          scale_units='xy', scale=1, width=0.012,
                                          label=lbl.split(' (')[0])
                        self._quiver_artists.append((q, lbl, raw))

            self._ax = ax
            ax.set_title(f'{len(self._paths)} path(s) — selected: {sel_p["name"]}',
                         color=self.ACCENT, fontsize=11, pad=10)
            ax.legend(loc='upper left', fontsize=7, facecolor=self.BG2,
                      edgecolor=self.FG, labelcolor=self.FG)
            self._fig.tight_layout()
            self._canvas.draw()
            self._update_info(sel_func, t0, sel_tmin, sel_tmax, sel_is_3d)
            self._hover_annotation = None
        except Exception as e:
            self._info_label.configure(text=f'Error: {e}')

    def _update_info(self, func, t0, tmin, tmax, is_3d):
        with np.errstate(divide='ignore', invalid='ignore'):
            pos = func(t0)
            v = numerical_derivative(func, t0)
            a = numerical_second_derivative(func, t0)
            speed = np.linalg.norm(v)
            T = tangent_vector(func, t0, normalize=True)
            N = normal_vector(func, t0)
            kappa = curvature(func, t0)
            tau = torsion(func, t0) if is_3d else np.nan

            def fmt(arr):
                if not np.all(np.isfinite(arr)): return 'Undefined'
                return '(' + ', '.join(f'{x:.4f}' for x in arr) + ')'

            self._kappa_label.configure(text=f'{kappa:.6f}' if np.isfinite(kappa) else 'Undefined')
            if is_3d:
                self._tau_label.configure(text=f'{tau:.6f}' if np.isfinite(tau) else 'Undefined')
            else:
                self._tau_label.configure(text='N/A (2D)')

            total_arc = arc_length(func, tmin, tmax, n_points=500)
            self._arclen_label.configure(text=f'{total_arc:.4f}' if np.isfinite(total_arc) else 'Undefined')

            lines = [f't₀ = {t0:.4f}', '',
                     f'r(t₀)     = {fmt(pos)}',
                     f"r'(t₀)    = {fmt(v)}",
                     f"|r'(t₀)|  = {speed:.4f}" if np.isfinite(speed) else "|r'(t₀)|  = Undefined",
                     f"r''(t₀)   = {fmt(a)}", '',
                     f'T          = {fmt(T)}',
                     f'N          = {fmt(N)}']

            if is_3d:
                B = binormal_vector(func, t0)
                lines += [f'B          = {fmt(B)}', '',
                          f'ρ (radius) = {1/kappa:.4f}' if np.isfinite(kappa) and kappa > 1e-8 else 'ρ (radius) = ∞']
            else:
                lines += ['', f'ρ (radius) = {1/kappa:.4f}' if np.isfinite(kappa) and kappa > 1e-8 else 'ρ (radius) = ∞']

            if np.isfinite(speed) and speed > 1e-10:
                aT = np.dot(v, a) / speed
                aN = (np.linalg.norm(np.cross(v, a)) / speed if is_3d
                      else np.sqrt(max(np.dot(a, a) - aT**2, 0)))
                lines += ['',
                          f'aT (tang)  = {aT:.4f}' if np.isfinite(aT) else 'aT (tang)  = Undefined',
                          f'aN (norm)  = {aN:.4f}' if np.isfinite(aN) else 'aN (norm)  = Undefined']

            arc_len = arc_length(func, tmin, t0, n_points=300)
            lines += ['', f'Arc len(0→t₀) = {arc_len:.4f}' if np.isfinite(arc_len) else 'Arc len = Undefined']
            self._info_label.configure(text='\n'.join(lines))

    # ── Hover tooltip ────────────────────────────────────────────────────

    def _on_hover(self, event):
        if event.inaxes is None or not hasattr(self, '_ax'):
            if self._hover_annotation is not None:
                self._hover_annotation.set_visible(False)
                self._canvas.draw_idle()
                self._hover_annotation = None
            return
        func = self._current_func
        if func is None:
            return
        t0 = self._t0_var.get()
        pos = func(t0)
        kappa_val = curvature(func, t0)
        hit_label = hit_raw = None
        for artist, label, raw in self._quiver_artists:
            try:
                if not self._is_3d:
                    if artist.contains(event)[0]:
                        hit_label, hit_raw = label, raw; break
                else:
                    sc = self._scale_var.get()
                    tip = pos + (raw*sc if any(k in label for k in ('Tangent','Normal','Binormal'))
                                 else raw*sc*0.5 if 'Velocity' in label else raw*sc*0.3)
                    from mpl_toolkits.mplot3d import proj3d
                    x2, y2, _ = proj3d.proj_transform(*tip, self._ax.get_proj())
                    disp = self._ax.transData.transform((x2, y2))
                    if np.linalg.norm(disp - np.array([event.x, event.y])) < 30:
                        hit_label, hit_raw = label, raw; break
            except Exception:
                continue
        if hit_label:
            fmt = lambda a: '(' + ', '.join(f'{v:.4f}' for v in a) + ')'
            tt = f'{hit_label}\nValue: {fmt(hit_raw)}\nPos: {fmt(pos)}\nκ = {kappa_val:.6f}'
            if self._is_3d:
                tt += f'\nτ = {torsion(func, t0):.6f}'
            if self._hover_annotation is None:
                self._hover_annotation = self._ax.annotate(
                    tt, xy=(event.xdata, event.ydata), xytext=(15, 15),
                    textcoords='offset points', fontsize=8, color=self.FG,
                    bbox=dict(boxstyle='round,pad=0.4', fc=self.BG2, ec=self.ACCENT, alpha=0.95),
                    arrowprops=dict(arrowstyle='->', color=self.ACCENT, lw=1.2))
            else:
                self._hover_annotation.set_text(tt)
                self._hover_annotation.xy = (event.xdata, event.ydata)
                self._hover_annotation.set_visible(True)
            self._canvas.draw_idle()
        elif self._hover_annotation is not None:
            self._hover_annotation.set_visible(False)
            self._canvas.draw_idle()
