/**
 * math_engine.js
 * JavaScript port of math_physics.py
 * All vector-calculus operations for vector-valued functions r(t).
 */

// ── Safe expression evaluator ──────────────────────────────────────────────
const SAFE_NS = {
  sin: Math.sin, cos: Math.cos, tan: Math.tan,
  arcsin: Math.asin, arccos: Math.acos, arctan: Math.atan,
  asin: Math.asin, acos: Math.acos, atan: Math.atan, atan2: Math.atan2,
  sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
  exp: Math.exp, log: Math.log, ln: Math.log, log2: Math.log2, log10: Math.log10,
  sqrt: Math.sqrt, abs: Math.abs, sign: Math.sign, pow: Math.pow,
  pi: Math.PI, e: Math.E, PI: Math.PI, E: Math.E,
  floor: Math.floor, ceil: Math.ceil, round: Math.round,
  min: Math.min, max: Math.max,
};

function safeEval(expr, t) {
  // Build function body with all safe names in scope
  const keys = Object.keys(SAFE_NS);
  const vals = keys.map(k => SAFE_NS[k]);
  // eslint-disable-next-line no-new-func
  const fn = new Function(...keys, 't', `"use strict"; return (${expr});`);
  return fn(...vals, t);
}

function makeScalarFn(expr) {
  const keys = Object.keys(SAFE_NS);
  const vals = keys.map(k => SAFE_NS[k]);
  const fn = new Function(...keys, 't', `"use strict"; return (${expr});`);
  return (t) => fn(...vals, t);
}

function makeVecFn(xExpr, yExpr, zExpr) {
  const fx = makeScalarFn(xExpr);
  const fy = makeScalarFn(yExpr);
  if (zExpr && zExpr.trim()) {
    const fz = makeScalarFn(zExpr);
    return { fn: (t) => [fx(t), fy(t), fz(t)], is3d: true };
  }
  return { fn: (t) => [fx(t), fy(t)], is3d: false };
}

// ── Linear algebra helpers ─────────────────────────────────────────────────
function norm(v) { return Math.sqrt(v.reduce((s, x) => s + x * x, 0)); }
function normalize(v) {
  const n = norm(v); if (n < 1e-12) return v.map(() => 0);
  return v.map(x => x / n);
}
function dot(a, b) { return a.reduce((s, x, i) => s + x * b[i], 0); }
function cross3(a, b) {
  return [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  ];
}
function add(a, b) { return a.map((x, i) => x + b[i]); }
function sub(a, b) { return a.map((x, i) => x - b[i]); }
function scale(a, s) { return a.map(x => x * s); }
function isFiniteVec(v) { return v.every(x => isFinite(x)); }

// ── Numerical calculus ─────────────────────────────────────────────────────
function derivative(fn, t, h = 1e-6) {
  const fwd = fn(t + h);
  const bwd = fn(t - h);
  return fwd.map((x, i) => (x - bwd[i]) / (2 * h));
}

function secondDerivative(fn, t, h = 1e-5) {
  const f0 = fn(t);
  const fwd = fn(t + h);
  const bwd = fn(t - h);
  return f0.map((x, i) => (fwd[i] - 2 * x + bwd[i]) / (h * h));
}

function tangentVector(fn, t, normalized = true) {
  const v = derivative(fn, t);
  return normalized ? normalize(v) : v;
}

function normalVector(fn, t) {
  const h = 1e-6;
  const T1 = tangentVector(fn, t - h, true);
  const T2 = tangentVector(fn, t + h, true);
  const dT = T1.map((x, i) => (T2[i] - x) / (2 * h));
  return normalize(dT);
}

function binormalVector(fn, t) {
  const T = tangentVector(fn, t, true);
  const N = normalVector(fn, t);
  if (T.length === 3) {
    const B = cross3(T, N);
    return normalize(B);
  }
  return [0, 0, 0];
}

function curvature(fn, t) {
  const h = 1e-6;
  const T1 = tangentVector(fn, t - h, true);
  const T2 = tangentVector(fn, t + h, true);
  const dT = T1.map((x, i) => (T2[i] - x) / (2 * h));
  const dT_norm = norm(dT);
  const v_norm = norm(derivative(fn, t));
  if (v_norm < 1e-12) return 0;
  return dT_norm / v_norm;
}

function torsion(fn, t) {
  const h = 1e-5;
  const r1 = derivative(fn, t);
  const r2 = secondDerivative(fn, t);
  const r2a = secondDerivative(fn, t + h);
  const r2b = secondDerivative(fn, t - h);
  const r3 = r2a.map((x, i) => (x - r2b[i]) / (2 * h));
  if (r1.length < 3) return NaN;
  const cr = cross3(r1, r2);
  const cr_norm_sq = dot(cr, cr);
  if (cr_norm_sq < 1e-12) return 0;
  return dot(cr, r3) / cr_norm_sq;
}

function arcLength(fn, tStart, tEnd, n = 500) {
  const dt = (tEnd - tStart) / (n - 1);
  let len = 0;
  for (let i = 0; i < n - 1; i++) {
    const t = tStart + i * dt;
    const v = derivative(fn, t);
    len += norm(v) * dt;
  }
  return len;
}

// ── Preset curves ──────────────────────────────────────────────────────────
const PRESETS = {
  'Helix':            { x: 'cos(t)', y: 'sin(t)', z: 't / 5',                     tmin: '0', tmax: '4*pi' },
  'Circle (2D)':      { x: 'cos(t)', y: 'sin(t)', z: '',                           tmin: '0', tmax: '2*pi' },
  'Trefoil Knot':     { x: 'sin(t)+2*sin(2*t)', y: 'cos(t)-2*cos(2*t)', z: '-sin(3*t)',  tmin: '0', tmax: '2*pi' },
  'Torus Knot (2,3)': { x: '(cos(3*t)+2)*cos(2*t)', y: '(cos(3*t)+2)*sin(2*t)', z: '-sin(3*t)', tmin: '0', tmax: '2*pi' },
  'Lissajous (2D)':   { x: 'sin(3*t+pi/2)', y: 'sin(2*t)', z: '',                tmin: '0', tmax: '2*pi' },
  'Cycloid (2D)':     { x: 't-sin(t)', y: '1-cos(t)', z: '',                      tmin: '0', tmax: '4*pi' },
  'Spiral (2D)':      { x: '0.1*t*cos(t)', y: '0.1*t*sin(t)', z: '',             tmin: '0', tmax: '6*pi' },
  'Viviani Curve':    { x: '1+cos(t)', y: 'sin(t)', z: '2*sin(t/2)',             tmin: '0', tmax: '2*pi' },
  'Parabolic 3D':     { x: 't', y: 't**2', z: 't**3',                            tmin: '-2', tmax: '2' },
};

// Analysis-tab presets (match Python analysis_tools.py)
const ANALYSIS_PRESETS = {
  'Helix':         { fn: t => [Math.cos(t), Math.sin(t), t*0.2],         tmin: 0,        tmax: 4*Math.PI, is3d: true },
  'Circle (2D)':   { fn: t => [Math.cos(t), Math.sin(t)],                tmin: 0,        tmax: 2*Math.PI, is3d: false },
  'Trefoil Knot':  { fn: t => [Math.sin(t)+2*Math.sin(2*t), Math.cos(t)-2*Math.cos(2*t), -Math.sin(3*t)], tmin: 0, tmax: 2*Math.PI, is3d: true },
  'Torus Knot':    { fn: t => { const r=Math.cos(3*t)+2; return [r*Math.cos(2*t),r*Math.sin(2*t),-Math.sin(3*t)]; }, tmin: 0, tmax: 2*Math.PI, is3d: true },
  'Lissajous (2D)':{ fn: t => [Math.sin(3*t+Math.PI/2), Math.sin(2*t)], tmin: 0,        tmax: 2*Math.PI, is3d: false },
  'Cycloid (2D)':  { fn: t => [t-Math.sin(t), 1-Math.cos(t)],            tmin: 0,        tmax: 4*Math.PI, is3d: false },
  'Spiral (2D)':   { fn: t => [0.1*t*Math.cos(t), 0.1*t*Math.sin(t)],   tmin: 0,        tmax: 6*Math.PI, is3d: false },
  'Parabola (3D)': { fn: t => [t, t*t, t*t*t],                           tmin: -2,       tmax: 2,         is3d: true },
};

function evalTRange(expr) {
  // evaluate expressions like "4*pi", "2*pi", "-2" etc.
  return safeEval(expr, 0);
}

function linspace(a, b, n) {
  const arr = [];
  for (let i = 0; i < n; i++) arr.push(a + (b - a) * i / (n - 1));
  return arr;
}
