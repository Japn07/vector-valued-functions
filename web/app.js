/**
 * app.js
 * Main UI logic for the Vector Calculus Multi-Tool App
 */

// ── Theme Management ────────────────────────────────────────────────────────
const themeToggle = document.getElementById('themeToggle');
const themeIcon = themeToggle.querySelector('.theme-icon');

function toggleTheme() {
  const isDark = document.body.classList.contains('dark');
  if (isDark) {
    document.body.classList.remove('dark');
    document.body.classList.add('light');
    themeIcon.textContent = '🌙';
  } else {
    document.body.classList.remove('light');
    document.body.classList.add('dark');
    themeIcon.textContent = '☀️';
  }
  updateChartsTheme();
  renderTNB();
}
themeToggle.addEventListener('click', toggleTheme);

// ── Tab Management ──────────────────────────────────────────────────────────
const tabs = document.querySelectorAll('.tab-btn');
const panes = document.querySelectorAll('.tab-pane');

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    tabs.forEach(t => { t.classList.remove('active'); t.setAttribute('aria-selected', 'false'); });
    panes.forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    tab.setAttribute('aria-selected', 'true');
    document.getElementById(`pane-${tab.dataset.tab}`).classList.add('active');
    
    // Trigger resize for Three.js and Chart.js to adapt
    window.dispatchEvent(new Event('resize'));
    if (tab.dataset.tab === 'tnb') renderTNB();
  });
});

// ============================================================================
// TAB 1 — TNB Visualizer
// ============================================================================

const PATH_COLORS = ['#e06c75', '#61afef', '#98c379', '#c678dd', '#e5c07b', '#56b6c2', '#be5046', '#d19a66', '#7ec8e3', '#c3e88d'];

let paths = [];
let selectedPathIdx = 0;
let is3D = true;
let currentFunc = null;

// UI Elements
const pathList = document.getElementById('pathList');
const btnAddPath = document.getElementById('btnAddPath');
const btnRemovePath = document.getElementById('btnRemovePath');
const btnUpdatePath = document.getElementById('btnUpdatePath');
const presetSelect = document.getElementById('presetSelect');
const btnPlot = document.getElementById('btnPlot');

const eX = document.getElementById('entryX');
const eY = document.getElementById('entryY');
const eZ = document.getElementById('entryZ');
const eTmin = document.getElementById('entryTmin');
const eTmax = document.getElementById('entryTmax');

const t0Slider = document.getElementById('t0Slider');
const t0Val = document.getElementById('t0Val');
const scaleSlider = document.getElementById('scaleSlider');
const scaleVal = document.getElementById('scaleVal');

const infoPanel = document.getElementById('infoPanel');
const errorBanner = document.getElementById('errorBanner');
const canvasTitle = document.getElementById('canvasTitle');
const canvasLegend = document.getElementById('canvasLegend');
const hoverTooltip = document.getElementById('hoverTooltip');

// Plotly setup
const plotlyContainer = document.getElementById('plotlyContainer');

function resizeTNB() {
  if (plotlyContainer && plotlyContainer.data) {
    Plotly.Plots.resize(plotlyContainer);
  }
}
window.addEventListener('resize', resizeTNB);
const resizeObserver = new ResizeObserver(resizeTNB);
resizeObserver.observe(plotlyContainer);

// ── UI Actions ──
function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.style.display = 'block';
}
function clearError() { errorBanner.style.display = 'none'; }

function updatePathListUI() {
  pathList.innerHTML = '';
  paths.forEach((p, i) => {
    const div = document.createElement('div');
    div.className = `path-item ${i === selectedPathIdx ? 'selected' : ''}`;
    const dot = document.createElement('div');
    dot.className = 'path-dot';
    dot.style.background = PATH_COLORS[i % PATH_COLORS.length];
    div.appendChild(dot);
    div.appendChild(document.createTextNode(`[${i+1}] ${p.name}`));
    div.addEventListener('click', () => {
      selectedPathIdx = i;
      loadPathToInputs(p);
      updatePathListUI();
      doPlot();
    });
    pathList.appendChild(div);
  });
}

function loadPathToInputs(p) {
  eX.value = p.x; eY.value = p.y; eZ.value = p.z;
  eTmin.value = p.tmin; eTmax.value = p.tmax;
}

function addPreset(name) {
  if (!name || !PRESETS[name]) return;
  const pre = PRESETS[name];
  paths.push({ x: pre.x, y: pre.y, z: pre.z, tmin: pre.tmin, tmax: pre.tmax, name });
  selectedPathIdx = paths.length - 1;
  loadPathToInputs(paths[selectedPathIdx]);
  updatePathListUI();
  presetSelect.value = '';
  doPlot();
}
presetSelect.addEventListener('change', (e) => addPreset(e.target.value));

btnAddPath.addEventListener('click', () => {
  const x = eX.value.trim(), y = eY.value.trim(), z = eZ.value.trim();
  if (!x || !y) return;
  const name = z ? `⟨${x}, ${y}, ${z}⟩` : `⟨${x}, ${y}⟩`;
  paths.push({ x, y, z, tmin: eTmin.value, tmax: eTmax.value, name });
  selectedPathIdx = paths.length - 1;
  updatePathListUI();
  doPlot();
});

btnRemovePath.addEventListener('click', () => {
  if (paths.length === 0) return;
  paths.splice(selectedPathIdx, 1);
  selectedPathIdx = Math.max(0, Math.min(selectedPathIdx, paths.length - 1));
  if (paths.length > 0) loadPathToInputs(paths[selectedPathIdx]);
  updatePathListUI();
  doPlot();
});

btnUpdatePath.addEventListener('click', () => {
  if (paths.length === 0) return;
  const x = eX.value.trim(), y = eY.value.trim(), z = eZ.value.trim();
  if (!x || !y) return;
  paths[selectedPathIdx] = { x, y, z, tmin: eTmin.value, tmax: eTmax.value, name: paths[selectedPathIdx].name };
  updatePathListUI();
  doPlot();
});

btnPlot.addEventListener('click', doPlot);
[eX, eY, eZ, eTmin, eTmax].forEach(el => {
  el.addEventListener('keydown', e => { if (e.key === 'Enter') doPlot(); });
});

t0Slider.addEventListener('input', () => {
  t0Val.textContent = parseFloat(t0Slider.value).toFixed(2);
  renderTNB();
});
scaleSlider.addEventListener('input', () => {
  scaleVal.textContent = parseFloat(scaleSlider.value).toFixed(2);
  renderTNB();
});

['showT','showN','showB','showV','showA'].forEach(id => {
  document.getElementById(id).addEventListener('change', renderTNB);
});

// ── Plotting Logic ──
function doPlot() {
  clearError();
  if (paths.length === 0) {
    clearScene();
    return;
  }
  try {
    is3D = paths.some(p => p.z.trim() !== '');
    const selP = paths[selectedPathIdx];
    const tmin = evalTRange(selP.tmin);
    const tmax = evalTRange(selP.tmax);
    
    t0Slider.min = tmin;
    t0Slider.max = tmax;
    let t0 = parseFloat(t0Slider.value);
    if (t0 < tmin || t0 > tmax || isNaN(t0)) {
      t0 = (tmin + tmax) / 2;
      t0Slider.value = t0;
      t0Val.textContent = t0.toFixed(2);
    }
    
    const { fn } = makeVecFn(selP.x, selP.y, selP.z);
    currentFunc = fn;
    
    renderTNB();
  } catch (err) {
    showError("Parse Error: " + err.message);
  }
}

function clearScene() {
  Plotly.purge('plotlyContainer');
  canvasTitle.textContent = '';
  infoPanel.textContent = '—';
}

function renderTNB() {
  if (paths.length === 0 || !currentFunc) return;
  clearError();
  
  const selP = paths[selectedPathIdx];
  const t0 = parseFloat(t0Slider.value);
  const scaleV = parseFloat(scaleSlider.value);
  const tmin = evalTRange(selP.tmin);
  const tmax = evalTRange(selP.tmax);
  
  canvasTitle.textContent = `${paths.length} path(s) — selected: ${selP.name}`;
  
  const isDark = document.body.classList.contains('dark');
  const fgColor = isDark ? '#cdd6f4' : '#4c4f69';
  const gridColor = isDark ? '#313244' : '#bcc0cc';
  
  let plotData = [];
  let minB = [Infinity, Infinity, Infinity], maxB = [-Infinity, -Infinity, -Infinity];
  
  // Evaluate all paths
  paths.forEach((p, i) => {
    try {
      const { fn } = makeVecFn(p.x, p.y, p.z);
      const tm = evalTRange(p.tmin), tx = evalTRange(p.tmax);
      const Tvals = linspace(tm, tx, 500);
      const pts = Tvals.map(t => fn(t)).filter(isFiniteVec);
      if (pts.length === 0) return;
      
      const color = PATH_COLORS[i % PATH_COLORS.length];
      const isSelected = i === selectedPathIdx;
      
      pts.forEach(v => {
        minB[0] = Math.min(minB[0], v[0]); maxB[0] = Math.max(maxB[0], v[0]);
        minB[1] = Math.min(minB[1], v[1]); maxB[1] = Math.max(maxB[1], v[1]);
        if(is3D) {
           const z = v[2]||0;
           minB[2] = Math.min(minB[2], z); maxB[2] = Math.max(maxB[2], z);
        }
      });
      
      if (is3D) {
        plotData.push({
          type: 'scatter3d',
          mode: 'lines',
          x: pts.map(v => v[0]),
          y: pts.map(v => v[1]),
          z: pts.map(v => v[2]||0),
          line: { color: color, width: isSelected ? 5 : 2 },
          opacity: isSelected ? 1 : 0.5,
          name: p.name,
          hoverinfo: 'none'
        });
      } else {
        plotData.push({
          type: 'scatter',
          mode: 'lines',
          x: pts.map(v => v[0]),
          y: pts.map(v => v[1]),
          line: { color: color, width: isSelected ? 3 : 1 },
          opacity: isSelected ? 1 : 0.5,
          name: p.name,
          hoverinfo: 'none'
        });
      }
    } catch(e) { console.error(e); }
  });
  
  if (plotData.length === 0) return;
  
  // Calculate vectors at t0
  let pos, v, a, T, N, B;
  try {
    pos = currentFunc(t0);
    v = derivative(currentFunc, t0);
    a = secondDerivative(currentFunc, t0);
    T = tangentVector(currentFunc, t0, true);
    N = normalVector(currentFunc, t0);
    if (is3D) B = binormalVector(currentFunc, t0);
    
    if (!isFiniteVec(pos)) throw new Error("Undefined position");
  } catch(e) {
    updateInfoText(null, t0, tmin, tmax);
    return;
  }
  
  // Plot current point
  if (is3D) {
    plotData.push({
      type: 'scatter3d', mode: 'markers',
      x: [pos[0]], y: [pos[1]], z: [pos[2]||0],
      marker: { color: '#f38ba8', size: 5 },
      name: 'Current Position', hoverinfo: 'x+y+z'
    });
  } else {
    plotData.push({
      type: 'scatter', mode: 'markers',
      x: [pos[0]], y: [pos[1]],
      marker: { color: '#f38ba8', size: 8 },
      name: 'Current Position', hoverinfo: 'x+y'
    });
  }

  // Draw Vectors
  const vecs = [];
  if (document.getElementById('showT').checked) vecs.push({ dir: T, color: '#3b82f6', name: 'T' });
  if (document.getElementById('showN').checked) vecs.push({ dir: N, color: '#22c55e', name: 'N' });
  if (document.getElementById('showB').checked && is3D) vecs.push({ dir: B, color: '#f97316', name: 'B' });
  if (document.getElementById('showV').checked) vecs.push({ dir: scale(v, 0.5), color: '#a855f7', name: 'v' });
  if (document.getElementById('showA').checked) vecs.push({ dir: scale(a, 0.3), color: '#06b6d4', name: 'a' });
  
  if (is3D) {
    vecs.forEach(vd => {
      if(!isFiniteVec(vd.dir)) return;
      const len = norm(vd.dir) * scaleV;
      if (len < 1e-6) return;
      
      const start = pos;
      const end = add(pos, scale(vd.dir, scaleV));
      
      // Calculate arrowhead (mimicking Matplotlib's quiver arrow_length_ratio=0.18)
      const arrowRatio = 0.18;
      const h = len * arrowRatio;
      const vNorm = normalize(vd.dir);
      
      // Find an orthogonal vector
      let u = cross3(vNorm, [1, 0, 0]);
      if (norm(u) < 1e-4) u = cross3(vNorm, [0, 1, 0]);
      u = normalize(u);
      
      const headBase = sub(end, scale(vNorm, h));
      const w = h * 0.35; // width of arrow head
      const tip1 = add(headBase, scale(u, w));
      const tip2 = sub(headBase, scale(u, w));
      
      // We can draw the arrow as a single line with NaN to break segments:
      // start -> end -> tip1 -> end -> tip2
      const xArr = [start[0], end[0], null, end[0], tip1[0], null, end[0], tip2[0]];
      const yArr = [start[1], end[1], null, end[1], tip1[1], null, end[1], tip2[1]];
      const zArr = [start[2]||0, end[2]||0, null, end[2]||0, tip1[2]||0, null, end[2]||0, tip2[2]||0];
      
      plotData.push({
        type: 'scatter3d',
        mode: 'lines',
        x: xArr, y: yArr, z: zArr,
        line: { color: vd.color, width: 4 },
        name: vd.name,
        hoverinfo: 'none'
      });
    });
  }
  
  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { l: 0, r: 0, t: 0, b: 0 },
    showlegend: true,
    legend: { x: 0, y: 1, font: { color: fgColor, family: 'Inter' }, bgcolor: 'transparent' },
    font: { color: fgColor, family: 'Inter' },
    uirevision: 'true' // Preserves zoom/pan/camera across slider updates
  };
  
  // Calculate padded ranges based on curves
  const pad = 1.2;
  const cx = (minB[0]+maxB[0])/2, rx = (maxB[0]-minB[0])/2 * pad;
  const cy = (minB[1]+maxB[1])/2, ry = (maxB[1]-minB[1])/2 * pad;
  const cz = (minB[2]+maxB[2])/2, rz = (maxB[2]-minB[2])/2 * pad;
  
  // Ensure we have a minimum range so flat curves don't break
  const rrx = Math.max(rx, 1e-3), rry = Math.max(ry, 1e-3), rrz = Math.max(rz, 1e-3);
  
  if (is3D) {
    layout.scene = {
      xaxis: { title: 'X', color: fgColor, gridcolor: gridColor, zerolinecolor: gridColor, showbackground: false, range: [cx - rrx, cx + rrx], autorange: false },
      yaxis: { title: 'Y', color: fgColor, gridcolor: gridColor, zerolinecolor: gridColor, showbackground: false, range: [cy - rry, cy + rry], autorange: false },
      zaxis: { title: 'Z', color: fgColor, gridcolor: gridColor, zerolinecolor: gridColor, showbackground: false, range: [cz - rrz, cz + rrz], autorange: false },
      aspectmode: 'data'
    };
  } else {
    layout.xaxis = { title: 'X', color: fgColor, gridcolor: gridColor, zerolinecolor: gridColor, range: [cx - rrx, cx + rrx], autorange: false };
    layout.yaxis = { title: 'Y', color: fgColor, gridcolor: gridColor, zerolinecolor: gridColor, scaleanchor: 'x', scaleratio: 1, range: [cy - rry, cy + rry], autorange: false };
    
    layout.annotations = [];
    vecs.forEach(vd => {
      if(!isFiniteVec(vd.dir)) return;
      const endPos = add(pos, scale(vd.dir, scaleV));
      layout.annotations.push({
        x: endPos[0], y: endPos[1],
        xref: 'x', yref: 'y',
        ax: pos[0], ay: pos[1],
        axref: 'x', ayref: 'y',
        showarrow: true,
        arrowcolor: vd.color,
        arrowsize: 1.5,
        arrowwidth: 2.5,
        arrowhead: 3
      });
    });
  }
  
  Plotly.react('plotlyContainer', plotData, layout, { responsive: true, displayModeBar: false });
  updateInfoText({ pos, v, a, T, N, B }, t0, tmin, tmax);
}

function updateInfoText(data, t0, tmin, tmax) {
  if (!data) {
    document.getElementById('kappaVal').textContent = '—';
    document.getElementById('tauVal').textContent = '—';
    document.getElementById('arcLenVal').textContent = '—';
    infoPanel.textContent = 'Undefined at this point';
    return;
  }
  const speed = norm(data.v);
  const kap = curvature(currentFunc, t0);
  const tau = is3D ? torsion(currentFunc, t0) : NaN;
  const totLen = arcLength(currentFunc, tmin, tmax);
  
  document.getElementById('kappaVal').textContent = isFinite(kap) ? kap.toFixed(5) : 'Undef';
  document.getElementById('tauVal').textContent = is3D ? (isFinite(tau) ? tau.toFixed(5) : 'Undef') : 'N/A';
  document.getElementById('arcLenVal').textContent = isFinite(totLen) ? totLen.toFixed(4) : 'Undef';
  
  const fmt = arr => isFiniteVec(arr) ? `(${arr.map(x=>x.toFixed(4)).join(', ')})` : 'Undefined';
  
  let lines = [
    `t₀ = ${t0.toFixed(4)}`, '',
    `r(t₀)     = ${fmt(data.pos)}`,
    `r'(t₀)    = ${fmt(data.v)}`,
    `|r'(t₀)|  = ${speed.toFixed(4)}`,
    `r''(t₀)   = ${fmt(data.a)}`, '',
    `T          = ${fmt(data.T)}`,
    `N          = ${fmt(data.N)}`,
  ];
  if (is3D && data.B) lines.push(`B          = ${fmt(data.B)}`);
  
  const rho = 1/kap;
  lines.push('');
  lines.push(`ρ (radius) = ${isFinite(rho) && rho < 1e8 ? rho.toFixed(4) : '∞'}`);
  
  if (speed > 1e-10) {
    const aT = dot(data.v, data.a) / speed;
    const aN = is3D ? norm(cross3(data.v, data.a))/speed : Math.sqrt(Math.max(dot(data.a, data.a) - aT*aT, 0));
    lines.push('', `aT (tang)  = ${aT.toFixed(4)}`, `aN (norm)  = ${aN.toFixed(4)}`);
  }
  
  const arcLenTo0 = arcLength(currentFunc, tmin, t0);
  lines.push('', `Arc len(0→t₀) = ${arcLenTo0.toFixed(4)}`);
  
  infoPanel.textContent = lines.join('\n');
}

// No custom hover tooltip needed for Plotly

// ============================================================================
// TAB 2 — Static Analysis Tools
// ============================================================================

const analysisCurveSelect = document.getElementById('analysisCurveSelect');
const customInputCard = document.getElementById('customInputCard');
const aEntryX = document.getElementById('aEntryX');
const aEntryY = document.getElementById('aEntryY');
const aEntryZ = document.getElementById('aEntryZ');
const aEntryTmin = document.getElementById('aEntryTmin');
const aEntryTmax = document.getElementById('aEntryTmax');
const analysisCharts = document.getElementById('analysisCharts');
const analysisError = document.getElementById('analysisError');

let chartInstances = [];

analysisCurveSelect.addEventListener('change', () => {
  if (analysisCurveSelect.value === 'Custom') {
    customInputCard.classList.remove('hidden');
  } else {
    customInputCard.classList.add('hidden');
  }
});
// Initialize state
customInputCard.classList.add('hidden');

function getAnalysisActiveFn() {
  const name = analysisCurveSelect.value;
  let fnObj, tmin, tmax;
  
  if (name === 'Custom') {
    fnObj = makeVecFn(aEntryX.value, aEntryY.value, aEntryZ.value);
    tmin = evalTRange(aEntryTmin.value);
    tmax = evalTRange(aEntryTmax.value);
  } else {
    const pre = ANALYSIS_PRESETS[name];
    fnObj = { fn: pre.fn, is3d: pre.is3d };
    tmin = pre.tmin;
    tmax = pre.tmax;
  }
  return { fn: fnObj.fn, is3D: fnObj.is3d, tmin, tmax, name };
}

function clearAnalysisCharts() {
  chartInstances.forEach(c => c.destroy());
  chartInstances = [];
  analysisCharts.innerHTML = '';
  analysisError.style.display = 'none';
}

function createChartContainer(title) {
  const card = document.createElement('div');
  card.className = 'chart-card';
  card.innerHTML = `<div class="chart-card-title">${title}</div><div class="chart-wrap"><canvas></canvas></div>`;
  analysisCharts.appendChild(card);
  return card.querySelector('canvas');
}

function getChartColors() {
  const isDark = document.body.classList.contains('dark');
  return {
    text: isDark ? '#cdd6f4' : '#4c4f69',
    grid: isDark ? '#313244' : '#bcc0cc',
    blue: isDark ? '#89b4fa' : '#1e66f5',
    green: isDark ? '#a6e3a1' : '#40a02b',
    red: isDark ? '#f38ba8' : '#d20f39',
    purple: isDark ? '#cba6f7' : '#8839ef',
    teal: isDark ? '#94e2d5' : '#179299'
  };
}

function updateChartsTheme() {
  const c = getChartColors();
  Chart.defaults.color = c.text;
  Chart.defaults.scale.grid.color = c.grid;
  chartInstances.forEach(chart => chart.update());
}

Chart.defaults.font.family = "'Inter', sans-serif";

document.getElementById('btnDerivatives').addEventListener('click', () => {
  try {
    const { fn, tmin, tmax, name } = getAnalysisActiveFn();
    clearAnalysisCharts();
    const tVals = linspace(tmin, tmax, 200);
    const d = getChartColors();
    
    // Components
    const dims = ['X', 'Y', 'Z'];
    ['Position r(t)', "Velocity r'(t)", "Acceleration r''(t)"].forEach((title, idx) => {
      const isPos = idx === 0, isVel = idx === 1;
      const dataX=[], dataY=[], dataZ=[];
      tVals.forEach(t => {
        let val;
        if(isPos) val = fn(t);
        else if(isVel) val = derivative(fn, t);
        else val = secondDerivative(fn, t);
        dataX.push({x:t, y:val[0]});
        dataY.push({x:t, y:val[1]});
        if(val.length > 2) dataZ.push({x:t, y:val[2]});
      });
      
      const datasets = [
        { label: 'X Component', data: dataX, borderColor: d.blue, fill: false, tension: 0.1, pointRadius: 0 },
        { label: 'Y Component', data: dataY, borderColor: d.green, fill: false, tension: 0.1, pointRadius: 0 }
      ];
      if (dataZ.length) {
        datasets.push({ label: 'Z Component', data: dataZ, borderColor: d.red, fill: false, tension: 0.1, pointRadius: 0 });
      }
      
      const canvas = createChartContainer(title);
      const c = new Chart(canvas, {
        type: 'line', data: { datasets },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: { x: { type: 'linear', title: { display: true, text: 't' } } }
        }
      });
      chartInstances.push(c);
    });
  } catch (err) { analysisError.textContent = err.message; analysisError.style.display = 'block'; }
});

document.getElementById('btnComponents').addEventListener('click', () => {
  try {
    const { fn, tmin, tmax, name } = getAnalysisActiveFn();
    clearAnalysisCharts();
    const tVals = linspace(tmin, tmax, 200);
    const d = getChartColors();
    
    const sample = fn(tVals[0]);
    const labels = sample.length > 2 ? ['x(t)', 'y(t)', 'z(t)'] : ['x(t)', 'y(t)'];
    const colors = [d.blue, d.green, d.red];
    
    labels.forEach((lbl, i) => {
      const data = tVals.map(t => ({ x: t, y: fn(t)[i] }));
      const canvas = createChartContainer(lbl);
      const c = new Chart(canvas, {
        type: 'line',
        data: { datasets: [{ label: lbl, data: data, borderColor: colors[i], fill: false, tension: 0.1, pointRadius: 0 }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'linear', title: { display: true, text: 't' } } } }
      });
      chartInstances.push(c);
    });
  } catch (err) { analysisError.textContent = err.message; analysisError.style.display = 'block'; }
});

document.getElementById('btnCurvaturePlot').addEventListener('click', () => {
  try {
    const { fn, tmin, tmax, name } = getAnalysisActiveFn();
    clearAnalysisCharts();
    const pad = (tmax - tmin) * 0.05;
    const tVals = linspace(tmin + pad, tmax - pad, 200);
    const data = tVals.map(t => ({ x: t, y: curvature(fn, t) }));
    const d = getChartColors();
    
    const canvas = createChartContainer(`Curvature κ along ${name}`);
    const c = new Chart(canvas, {
      type: 'line',
      data: { datasets: [{ label: 'Curvature κ', data: data, borderColor: d.purple, backgroundColor: `${d.purple}33`, fill: true, tension: 0.1, pointRadius: 0 }] },
      options: { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'linear', title: { display: true, text: 't' } } } }
    });
    chartInstances.push(c);
  } catch (err) { analysisError.textContent = err.message; analysisError.style.display = 'block'; }
});

document.getElementById('btnArcLength').addEventListener('click', () => {
  try {
    const { fn, tmin, tmax, name } = getAnalysisActiveFn();
    clearAnalysisCharts();
    const tVals = linspace(tmin, tmax, 50);
    const data = tVals.map(t => ({ x: t, y: arcLength(fn, tmin, t) }));
    const d = getChartColors();
    
    const canvas = createChartContainer(`Arc Length s(t) for ${name}`);
    const c = new Chart(canvas, {
      type: 'line',
      data: { datasets: [{ label: 'Arc Length s(t)', data: data, borderColor: d.teal, backgroundColor: `${d.teal}33`, fill: true, tension: 0.1, pointRadius: 0 }] },
      options: { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'linear', title: { display: true, text: 't' } } } }
    });
    chartInstances.push(c);
  } catch (err) { analysisError.textContent = err.message; analysisError.style.display = 'block'; }
});

// Init
addPreset('Helix');
