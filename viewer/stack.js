// Stack preview
// Shows each sculpture option exported by slice_layers.py --viewer as it will
// be built (solid layers of real sheet thickness), next to the visualizer's
// contour stack drawn at the model's own proportions. Every panel uses the
// same camera and the same scale, so sizes and proportions compare directly.
//
// Data
//   stacks/index.json          { "stacks": ["name", ...] }
//   stacks/<name>/stack.stl    the layered object, millimetres, z up
//   stacks/<name>/info.json    size, layers, thickness, frame, sheets
//   data/frames.json           the visualizer's contours (reference panel)

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// ---------------------------------------------------------------------------
// 1. Configuration
// ---------------------------------------------------------------------------
const CONFIG = {
  framesUrl: 'data/frames.json',
  stacksUrl: 'stacks/index.json',
  reference: true,          // show the visualizer's contour stack as the first panel
  camera: { elevation: 12, azimuth: 35, margin: 1.12 },   // same view as index.html
  autoRotate: 0.35,
  colors: {                 // the visualizer's palette
    fill: 0xd8692e,
    line: 0xf2b48a,
    acrylic: 0xeaf7f0,      // clear sheet, very slightly green
    acrylicEdge: 0x4fae84,  // the green you see on cut edges
  },
};

// ---------------------------------------------------------------------------
// 2. Renderer, camera, controls (one canvas, one viewport per panel)
// ---------------------------------------------------------------------------
const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setScissorTest(true);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
const envMap = new THREE.PMREMGenerator(renderer)
  .fromScene(new RoomEnvironment(), 0.04).texture;

const panelsEl = document.getElementById('panels');
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -2000, 2000);
const controls = new OrbitControls(camera, panelsEl);
controls.enableDamping = true;
controls.enablePan = false;
controls.autoRotate = CONFIG.autoRotate > 0;
controls.autoRotateSpeed = CONFIG.autoRotate;

const e0 = THREE.MathUtils.degToRad(CONFIG.camera.elevation);
const a0 = THREE.MathUtils.degToRad(CONFIG.camera.azimuth);
camera.position.set(500 * Math.cos(e0) * Math.sin(a0), 500 * Math.sin(e0),
  500 * Math.cos(e0) * Math.cos(a0));
controls.target.set(0, 0, 0);
camera.lookAt(0, 0, 0);

// ---------------------------------------------------------------------------
// 3. Loading
// ---------------------------------------------------------------------------
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not load ${url} (${r.status})`);
  return r.json();
}

function fail(msg) {
  panelsEl.innerHTML = `<p id="message">${msg}</p>`;
  throw new Error(msg);
}

let names = [];
try {
  names = (await getJSON(CONFIG.stacksUrl)).stacks;
} catch {
  fail('No stacks found. Run slice_layers.py with --viewer pointing at this folder, ' +
       'e.g. <code>--viewer viewer --name thick</code>, then reload.');
}
const stacks = await Promise.all(names.map(async (name) => {
  const info = await getJSON(`stacks/${name}/info.json`);
  const buf = await (await fetch(`stacks/${name}/stack.stl`)).arrayBuffer();
  return { info, geometry: new STLLoader().parse(buf) };
}));
let frames = null;
if (CONFIG.reference) {
  try { frames = await getJSON(CONFIG.framesUrl); } catch { frames = null; }
}

// ---------------------------------------------------------------------------
// 4. Geometry in a shared frame of reference
// ---------------------------------------------------------------------------
// Scene axes: x = east, y = up, z = south, in millimetres. The slicer works
// in the model's own frame (x east from the west frame edge, y north from the
// south edge, z up), so a stack and the contours line up when both use it.
// Each panel is centred on the middle of its footprint and of its height.
const frame = stacks.find((s) => s.info.frame)?.info.frame;

function stlToScene(g, depth) {
  g.rotateX(-Math.PI / 2);             // (x, y, z) -> (x, z, -y)
  g.translate(0, 0, depth);            // north edge at z = 0, south at z = depth
  return g;
}

const fmtIn = (mm) => (mm / 25.4).toFixed(1);
const sizeText = (s) =>
  `${fmtIn(s[0])} × ${fmtIn(s[1])} × ${fmtIn(s[2])} in ` +
  `(${s.map((v) => Math.round(v)).join(' × ')} mm)`;

// Materials, two looks
const mats = {
  visualizer: {
    body: new THREE.MeshStandardMaterial({ color: CONFIG.colors.fill, roughness: 0.85,
      flatShading: true, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }),
    edge: new THREE.LineBasicMaterial({ color: CONFIG.colors.line }),
  },
  acrylic: {
    body: new THREE.MeshPhysicalMaterial({ color: CONFIG.colors.acrylic, metalness: 0,
      roughness: 0.05, transmission: 1, thickness: 5, ior: 1.49,
      attenuationColor: new THREE.Color(CONFIG.colors.acrylicEdge), attenuationDistance: 40,
      envMap, envMapIntensity: 1.2, flatShading: true, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }),
    edge: new THREE.LineBasicMaterial({ color: CONFIG.colors.acrylicEdge,
      transparent: true, opacity: 0.6 }),
  },
};
const refMats = {
  fill: new THREE.MeshBasicMaterial({ color: CONFIG.colors.fill, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }),
  line: new THREE.LineBasicMaterial({ color: CONFIG.colors.line }),
};

function addLights(scene) {
  scene.add(new THREE.HemisphereLight(0xffffff, 0x303030, 1.4));
  const d = new THREE.DirectionalLight(0xffffff, 1.6);
  d.position.set(-120, 300, 180);
  scene.add(d);
}

const panels = [];

function addPanel(title, detail, scene, center, extent) {
  const el = document.createElement('section');
  el.className = 'panel';
  el.innerHTML = `<div class="label"><b>${title}</b>${detail.map((d) => `<span>${d}</span>`).join('')}</div>`;
  panelsEl.appendChild(el);
  scene.position.set(-center.x, -center.y, -center.z);
  const root = new THREE.Scene();
  root.add(scene);
  panels.push({ el, root, extent });
}

// 4a. The visualizer's contour stack at the model's own proportions
function signedArea(p) {
  let a = 0;
  for (let i = 0, j = p.length - 1; i < p.length; j = i++) a += (p[j][0] + p[i][0]) * (p[j][1] - p[i][1]);
  return a / 2;
}
function inside([x, y], p) {
  let r = false;
  for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
    const [xi, yi] = p[i], [xj, yj] = p[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) r = !r;
  }
  return r;
}
function shapesFrom(contours, toV) {
  const items = contours.map((c) => ({ c, area: Math.abs(signedArea(c)) }))
    .sort((a, b) => b.area - a.area);
  for (const it of items) {
    const outer = items.filter((o) => o !== it && o.area > it.area && inside(it.c[0], o.c));
    it.depth = outer.length;
    it.parent = outer.length ? outer[outer.length - 1] : null;
  }
  const shapes = new Map();
  for (const it of items) if (it.depth % 2 === 0) shapes.set(it, new THREE.Shape(it.c.map(toV)));
  for (const it of items) {
    if (it.depth % 2 === 1 && shapes.has(it.parent)) {
      shapes.get(it.parent).holes.push(new THREE.Path(it.c.map(toV)));
    }
  }
  return [...shapes.values()];
}

if (frames && frame) {
  // Only outlines inside the stacks' footprint, so small cells far out in the
  // frame (left out of the sculpture) do not set the panel size.
  const box = new THREE.Box3();
  for (const s of stacks) {
    const g = s.geometry.clone();
    stlToScene(g, frame.depth);
    g.computeBoundingBox();
    box.union(g.boundingBox);
  }
  const W = frame.width, D = frame.depth, H = frame.height;
  const group = new THREE.Group();
  const pad = 0.04 * W;
  for (const f of frames.frames) {
    const y = f.z * H;
    const kept = f.contours.filter((c) => c.every(([u, v]) =>
      u * W >= box.min.x - pad && u * W <= box.max.x + pad &&
      v * D >= box.min.z - pad && v * D <= box.max.z + pad));
    for (const c of kept) {
      const pos = new Float32Array(c.length * 3);
      c.forEach(([u, v], k) => pos.set([u * W, y, v * D], k * 3));
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      group.add(new THREE.LineLoop(g, refMats.line));
    }
    if (kept.length) {
      const fill = new THREE.Mesh(
        new THREE.ShapeGeometry(shapesFrom(kept, ([u, v]) => new THREE.Vector2(u * W, v * D))),
        refMats.fill);
      fill.rotation.x = Math.PI / 2;    // (x, y) -> (x, 0, y)
      fill.position.y = y;
      group.add(fill);
    }
  }
  const c = new THREE.Vector3((box.min.x + box.max.x) / 2, H / 2, (box.min.z + box.max.z) / 2);
  const size = [box.max.x - box.min.x, box.max.z - box.min.z, H];
  addPanel('Visualizer', [`${frames.frames.length} frames, model proportions`,
    `${sizeText(size)} at this width`], group, c,
    { r: Math.hypot(size[0], size[1]) / 2, h: H });
}

// 4b. Each option as built
const stepMin = frames && frames.frames.length > 1
  ? Math.round((new Date(frames.frames[1].time) - new Date(frames.frames[0].time)) / 60000) : 0;
for (const s of stacks) {
  const info = s.info;
  const g = stlToScene(s.geometry, frame ? frame.depth : 0);
  g.computeBoundingBox();
  const bb = g.boundingBox;
  const body = new THREE.Mesh(g, mats.visualizer.body);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(g, 25), mats.visualizer.edge);
  const group = new THREE.Group();
  group.add(body, edges);
  addLights(group);
  const c = new THREE.Vector3((bb.min.x + bb.max.x) / 2, (bb.min.y + bb.max.y) / 2,
    (bb.min.z + bb.max.z) / 2);
  const detail = [
    `${info.layers} layers × ${info.thickness} mm` +
      (info.every ? `, every ${info.every * stepMin || info.every + ' frames'}` +
        `${stepMin ? ' min' : ''} (${info.first}–${info.last})` : ''),
    sizeText(info.size),
  ];
  if (frame) {
    const ratio = info.size[2] / info.size[0], model = frame.height / frame.width;
    detail.push(`${(ratio / model).toFixed(1)}× the visualizer's height for its width` +
      (info.sheets ? ` · ${info.sheets} sheet${info.sheets > 1 ? 's' : ''}` : ''));
  }
  addPanel(info.label, detail, group, c,
    { r: Math.hypot(bb.max.x - bb.min.x, bb.max.z - bb.min.z) / 2, h: bb.max.y - bb.min.y });
  s.body = body;
  s.edges = edges;
}

// ---------------------------------------------------------------------------
// 5. Controls: material and rotation
// ---------------------------------------------------------------------------
document.querySelectorAll('button[data-mode]').forEach((b) => b.addEventListener('click', () => {
  const m = mats[b.dataset.mode];
  for (const s of stacks) { s.body.material = m.body; s.edges.material = m.edge; }
  if (still) render();
  document.querySelectorAll('button[data-mode]').forEach((o) =>
    o.setAttribute('aria-pressed', String(o === b)));
}));
const spin = document.getElementById('spin');
spin.addEventListener('click', () => {
  controls.autoRotate = !controls.autoRotate;
  spin.setAttribute('aria-pressed', String(controls.autoRotate));
});

// ---------------------------------------------------------------------------
// 6. Render: one shared scale, so the largest option fills its panel and the
// others appear at their true relative size
// ---------------------------------------------------------------------------
const R = Math.max(...panels.map((p) => p.extent.r));
const Hmax = Math.max(...panels.map((p) => p.extent.h));

function render() {
  controls.update();
  const w = window.innerWidth, h = window.innerHeight;
  if (canvas.width !== Math.floor(w * renderer.getPixelRatio())) renderer.setSize(w, h, false);
  renderer.setScissor(0, 0, w, h);
  renderer.clear();
  const e = Math.asin(Math.min(1, Math.max(-1,
    (camera.position.y - controls.target.y) / camera.position.distanceTo(controls.target))));
  for (const p of panels) {
    const r = p.el.getBoundingClientRect();
    if (r.bottom < 0 || r.top > h || r.width < 2 || r.height < 2) continue;
    const top = r.top + 56, ph = r.height - 56;          // leave room for the label
    const aspect = r.width / ph;
    const halfV = (Hmax / 2) * Math.cos(e) + R * Math.abs(Math.sin(e));
    const fh = 2 * Math.max(halfV, R / aspect) * CONFIG.camera.margin;
    camera.left = (-fh * aspect) / 2; camera.right = (fh * aspect) / 2;
    camera.top = fh / 2; camera.bottom = -fh / 2;
    camera.updateProjectionMatrix();
    renderer.setViewport(r.left, h - top - ph, r.width, ph);
    renderer.setScissor(r.left, h - top - ph, r.width, ph);
    renderer.render(p.root, camera);
  }
  if (!still) requestAnimationFrame(render);
}
// ?still renders a single frame (for screenshots or slow machines)
const still = new URLSearchParams(location.search).has('still');
render();
window.stackPreview = { panels, controls, ready: true };
