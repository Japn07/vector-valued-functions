import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
# =============================================================================
# Vector-Valued Function Definitions
# =============================================================================

def helix(t, a=1, b=0.2):
    """3D Helix: r(t) = (a*cos(t), a*sin(t), b*t)"""
    x = a * np.cos(t)
    y = a * np.sin(t)
    z = b * t
    return np.array([x, y, z])

def circle_2d(t, r=1):
    """2D Circle: r(t) = (r*cos(t), r*sin(t))"""
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.array([x, y])

def lissajous_2d(t, a=3, b=2, delta=np.pi/2):
    """2D Lissajous curve: r(t) = (sin(a*t + delta), sin(b*t))"""
    x = np.sin(a * t + delta)
    y = np.sin(b * t)
    return np.array([x, y])

def spiral_2d(t, a=0.1):
    """2D Spiral: r(t) = (a*t*cos(t), a*t*sin(t))"""
    x = a * t * np.cos(t)
    y = a * t * np.sin(t)
    return np.array([x, y])

def trefoil_knot(t):
    """3D Trefoil Knot"""
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    return np.array([x, y, z])

def torus_knot(t, p=2, q=3):
    """3D Torus Knot"""
    r = np.cos(q * t) + 2
    x = r * np.cos(p * t)
    y = r * np.sin(p * t)
    z = -np.sin(q * t)
    return np.array([x, y, z])

def parabola_3d(t):
    """3D Parabolic path: r(t) = (t, t^2, t^3)"""
    x = t
    y = t ** 2
    z = t ** 3
    return np.array([x, y, z])

def cycloid_2d(t, r=1):
    """2D Cycloid: path traced by point on rolling circle"""
    x = r * (t - np.sin(t))
    y = r * (1 - np.cos(t))
    return np.array([x, y])


# =============================================================================
# Calculus Operations
# =============================================================================

def numerical_derivative(func, t, h=1e-6):
    """
    Compute the derivative of a vector-valued function at point t.
    Uses central difference for better accuracy.
    """
    return (func(t + h) - func(t - h)) / (2 * h)

def numerical_second_derivative(func, t, h=1e-5):
    """
    Compute the second derivative of a vector-valued function at point t.
    """
    return (func(t + h) - 2 * func(t) + func(t - h)) / (h ** 2)

def tangent_vector(func, t, normalize=True):
    """
    Compute the tangent vector (velocity) at point t.
    T = r'(t) / |r'(t)| if normalized
    """
    v = numerical_derivative(func, t)
    if normalize:
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            return v / norm
    return v

def normal_vector(func, t):
    """
    Compute the principal normal vector at point t.
    N = T'(t) / |T'(t)|
    """
    h = 1e-6
    T1 = tangent_vector(func, t - h, normalize=True)
    T2 = tangent_vector(func, t + h, normalize=True)
    dT = (T2 - T1) / (2 * h)
    norm = np.linalg.norm(dT)
    if norm > 1e-10:
        return dT / norm
    return np.zeros_like(dT)

def binormal_vector(func, t):
    """
    Compute the binormal vector at point t (3D only).
    B = T × N
    """
    T = tangent_vector(func, t, normalize=True)
    N = normal_vector(func, t)
    if len(T) == 3:
        B = np.cross(T, N)
        norm = np.linalg.norm(B)
        if norm > 1e-10:
            return B / norm
    return np.zeros(3)

def curvature(func, t):
    """
    Compute the curvature κ at point t.
    κ = |T'(t)| / |r'(t)|
    """
    h = 1e-6
    T1 = tangent_vector(func, t - h, normalize=True)
    T2 = tangent_vector(func, t + h, normalize=True)
    dT_norm = np.linalg.norm((T2 - T1) / (2 * h))
    v_norm = np.linalg.norm(numerical_derivative(func, t))
    if v_norm > 1e-10:
        return dT_norm / v_norm
    return 0

def torsion(func, t):
    """
    Compute the torsion τ at point t (3D only).
    τ = (r' × r'') · r''' / |r' × r''|²
    """
    h = 1e-5
    r1 = numerical_derivative(func, t)
    r2 = numerical_second_derivative(func, t)
    # Third derivative via central difference of second derivative
    r3 = (numerical_second_derivative(func, t + h) - numerical_second_derivative(func, t - h)) / (2 * h)
    
    if len(r1) == 3:
        cross = np.cross(r1, r2)
        cross_norm_sq = np.dot(cross, cross)
        if cross_norm_sq > 1e-10:
            return np.dot(cross, r3) / cross_norm_sq
    return 0.0

def arc_length(func, t_start, t_end, n_points=1000):
    """
    Compute the arc length of the curve from t_start to t_end.
    L = ∫|r'(t)| dt
    """
    t_vals = np.linspace(t_start, t_end, n_points)
    dt = (t_end - t_start) / (n_points - 1)
    
    length = 0
    for t in t_vals[:-1]:
        v = numerical_derivative(func, t)
        length += np.linalg.norm(v) * dt
    return length

def line_integral_scalar(func, scalar_field, t_start, t_end, n_points=1000):
    """
    Compute line integral of scalar field f along curve r(t).
    ∫f(r(t)) |r'(t)| dt
    """
    t_vals = np.linspace(t_start, t_end, n_points)
    dt = (t_end - t_start) / (n_points - 1)
    
    integral = 0
    for t in t_vals[:-1]:
        pos = func(t)
        v = numerical_derivative(func, t)
        speed = np.linalg.norm(v)
        integral += scalar_field(*pos) * speed * dt
    return integral

def line_integral_vector(func, vector_field, t_start, t_end, n_points=1000):
    """
    Compute line integral of vector field F along curve r(t).
    ∫F(r(t)) · r'(t) dt
    """
    t_vals = np.linspace(t_start, t_end, n_points)
    dt = (t_end - t_start) / (n_points - 1)
    
    integral = 0
    for t in t_vals[:-1]:
        pos = func(t)
        v = numerical_derivative(func, t)
        F = vector_field(*pos)
        integral += np.dot(F, v) * dt
    return integral


