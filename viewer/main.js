// Storm volume viewer
// Reads the frames.json and textures/ written by build_volume.py and shows
// the storm as a stack of traced contours, read through time by a moving
// image plane, with a section (slit-scan) view below.
//
// The file is organised in the order you would build it:
//   1. Configuration          5. Image plane (the playhead)
//   2. Scene and camera       6. Timestamp label
//   3. Loading the data       7. Elevation view (lower panel)
//   4. Contour stack          8. Timeline, controls and the render loop

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ---------------------------------------------------------------------------
// 1. Configuration
// ---------------------------------------------------------------------------
const CONFIG = {
  dataUrl: "data/frames.json",
  width: 10, // west-east extent in scene units
  heightRatio: 0.37, // time-axis height relative to width (sculpture: 69/189)
  secondsPerFrame: 0.45, // playback speed
  holdAtEnd: 1.6, // seconds to pause on the last frame before looping
  autoRotate: 0.35, // camera orbit speed (0 to disable)
  camera: {
    elevation: 12, // degrees above the horizontal; lower = more side-on
    azimuth: 35, // degrees around the vertical axis (0 = from the south)
    zoom: 1.0, // >1 zooms in further (edges may crop while orbiting)
    margin: 1.03, // breathing room around the volume when fitted
  },
  section: {
    width: 0.42, // share of the available width used by the drawing
    height: 0.72, // share of the available height used by the drawing
    lightUp: true, // rows light up in sync with playback, using the same
    // colours as the volume above (past, current, future)
  },
  timestamp: {
    mode: "rise", // 'rise': fixed at the left, rising with the image plane
    // 'fixed': completely still at one screen position
    left: 24, // distance from the left edge of the view [px]
    top: 24, // used only in 'fixed' mode [px]
  },
  colors: {
    background: 0x000000,
    pastFill: 0x2e5bd8,
    pastLine: 0xf58319,
    futureLine: 0xf58319,
    plane: 0xffffff,
  },
};

// ---------------------------------------------------------------------------
// 2. Scene and camera
// ---------------------------------------------------------------------------
const view = document.getElementById("view3d");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(CONFIG.colors.background);

// An orthographic camera keeps parallel lines parallel, which gives the
// flat, axonometric look of an architectural drawing.
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -100, 100);

// The view is fitted to the volume so that it fills the frame. While the
// camera orbits, the widest the volume can appear is its half-diagonal R, so
// fitting to R keeps it in view at every angle without cropping.
let fitR = 6,
  fitH = 4;
function frustumHeight(aspect) {
  const e = THREE.MathUtils.degToRad(CONFIG.camera.elevation);
  const halfV = (fitH / 2) * Math.cos(e) + fitR * Math.sin(e);
  return 2 * Math.max(halfV, fitR / aspect) * CONFIG.camera.margin;
}
function placeCamera(target) {
  const e = THREE.MathUtils.degToRad(CONFIG.camera.elevation);
  const a = THREE.MathUtils.degToRad(CONFIG.camera.azimuth);
  camera.position.set(
    target.x + 30 * Math.cos(e) * Math.sin(a),
    target.y + 30 * Math.sin(e),
    target.z + 30 * Math.cos(e) * Math.cos(a),
  );
  camera.zoom = CONFIG.camera.zoom;
  camera.lookAt(target);
}

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = CONFIG.autoRotate > 0;
controls.autoRotateSpeed = CONFIG.autoRotate;
controls.enablePan = false;

function resize() {
  const w = view.clientWidth,
    h = view.clientHeight;
  renderer.setSize(w, h, false);
  const aspect = w / Math.max(h, 1);
  const fh = frustumHeight(aspect);
  camera.left = (-fh * aspect) / 2;
  camera.right = (fh * aspect) / 2;
  camera.top = fh / 2;
  camera.bottom = -fh / 2;
  camera.updateProjectionMatrix();
  drawSection();
}
window.addEventListener("resize", resize);

// ---------------------------------------------------------------------------
// 3. Loading the data
// ---------------------------------------------------------------------------
// frames.json: { threshold_K, aspect, frames: [ { time, label, z, texture,
//   contours: [ [[u, v], ...], ... ] } ] }
// u runs west -> east and v north -> south, both from 0 to 1. z runs 0 -> 1
// from the first frame to the last.
const meta = await fetch(CONFIG.dataUrl).then((r) => {
  if (!r.ok) throw new Error(`Could not load ${CONFIG.dataUrl} (${r.status})`);
  return r.json();
});
const frames = meta.frames;
const N = frames.length;
const W = CONFIG.width;
const D = W * meta.aspect; // north-south extent
const H = W * CONFIG.heightRatio; // time axis

// Scene axes: x = east, z = south, y = time (up).
const toX = (u) => (u - 0.5) * W;
const toZ = (v) => (v - 0.5) * D;

const day = new Date(frames[0].time);
document.getElementById("title").textContent =
  `${day.toLocaleDateString("en-CA", { day: "numeric", month: "long", year: "numeric" })}`;
document.getElementById("subtitle").textContent =
  `Cloud tops colder than ${meta.threshold_K} K over southern Ontario, ` +
  `${frames[0].label}–${frames[N - 1].label}`;

fitR = Math.hypot(W, D) / 2;
fitH = H;
controls.target.set(0, H / 2, 0);
placeCamera(controls.target);

// ---------------------------------------------------------------------------
// 4. Contour stack
// ---------------------------------------------------------------------------
// Each frame becomes a group holding its outlines (LineLoop) and, for frames
// already passed, a filled version of the same shapes.
const materials = {
  pastLine: new THREE.LineBasicMaterial({ color: CONFIG.colors.pastLine }),
  futureLine: new THREE.LineBasicMaterial({
    color: CONFIG.colors.futureLine,
    transparent: true,
    opacity: 0.55,
  }),
  // Solid fills; the polygon offset keeps the outlines drawn on top of them.
  pastFill: new THREE.MeshBasicMaterial({
    color: CONFIG.colors.pastFill,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  }),
};

function signedArea(pts) {
  let a = 0;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    a += (pts[j][0] + pts[i][0]) * (pts[j][1] - pts[i][1]);
  }
  return a / 2;
}

function pointInPolygon([x, y], pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i],
      [xj, yj] = pts[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

// Contours that sit inside an odd number of other contours are holes (clear
// sky enclosed by cloud). Each hole is attached to the smallest outer
// contour that contains it.
function buildShapes(contours) {
  const items = contours
    .map((c) => ({ c, area: Math.abs(signedArea(c)) }))
    .sort((a, b) => b.area - a.area);
  for (const it of items) {
    const containers = items.filter(
      (o) => o !== it && o.area > it.area && pointInPolygon(it.c[0], o.c),
    );
    it.depth = containers.length;
    it.parent = containers.length ? containers[containers.length - 1] : null;
  }
  const shapes = new Map();
  const toVec = (p) => new THREE.Vector2(toX(p[0]), toZ(p[1]));
  for (const it of items) {
    if (it.depth % 2 === 0) shapes.set(it, new THREE.Shape(it.c.map(toVec)));
  }
  for (const it of items) {
    if (it.depth % 2 === 1 && shapes.has(it.parent)) {
      shapes.get(it.parent).holes.push(new THREE.Path(it.c.map(toVec)));
    }
  }
  return [...shapes.values()];
}

const layers = frames.map((f, i) => {
  const y = f.z * H;
  const group = new THREE.Group();
  const lines = f.contours.map((c) => {
    const pos = new Float32Array(c.length * 3);
    c.forEach(([u, v], k) => pos.set([toX(u), y, toZ(v)], k * 3));
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const line = new THREE.LineLoop(geom, materials.futureLine);
    group.add(line);
    return line;
  });
  let fill = null;
  if (f.contours.length) {
    // ShapeGeometry lies in the XY plane; rotating +90 degrees about X maps
    // (x, y) to (x, 0, y), i.e. onto the horizontal plane with y as south.
    fill = new THREE.Mesh(
      new THREE.ShapeGeometry(buildShapes(f.contours)),
      materials.pastFill,
    );
    fill.rotation.x = Math.PI / 2;
    fill.position.y = y;
    fill.visible = false;
    group.add(fill);
  }
  scene.add(group);
  return { lines, fill, index: i };
});

function setLayerStates(t) {
  for (const L of layers) {
    const past = L.index <= t + 1e-6;
    for (const line of L.lines) {
      line.material = past ? materials.pastLine : materials.futureLine;
    }
    if (L.fill) L.fill.visible = past;
  }
}

// A faint outline of the frame at the base and top of the volume.
function frameOutline(y, opacity) {
  const pts = [
    [-W / 2, -D / 2],
    [W / 2, -D / 2],
    [W / 2, D / 2],
    [-W / 2, D / 2],
  ].map(([x, z]) => new THREE.Vector3(x, y, z));
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  return new THREE.LineLoop(
    g,
    new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity,
    }),
  );
}
scene.add(frameOutline(0, 0.25), frameOutline(H, 0.12));

// ---------------------------------------------------------------------------
// 5. Image plane (the playhead)
// ---------------------------------------------------------------------------
// A horizontal plane carrying the satellite image of the current moment.
const loader = new THREE.TextureLoader();
const dataBase = new URL(CONFIG.dataUrl, window.location.href); // textures sit beside frames.json
const textures = frames.map((f) => {
  const tex = loader.load(new URL(f.texture, dataBase).href);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
});

const plane = new THREE.Mesh(
  new THREE.PlaneGeometry(W, D),
  // Fully opaque: the satellite image is shown as recorded.
  new THREE.MeshBasicMaterial({
    map: textures[0],
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: 2,
    polygonOffsetUnits: 2,
  }),
);
// PlaneGeometry lies in XY with the image's top edge at +y; rotating -90
// degrees about X puts that edge at -z, which is north in this scene.
plane.rotation.x = -Math.PI / 2;
scene.add(plane);

const planeEdge = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.PlaneGeometry(W, D)),
  new THREE.LineBasicMaterial({
    color: CONFIG.colors.plane,
    transparent: true,
    opacity: 0.6,
  }),
);
planeEdge.rotation.x = -Math.PI / 2;
scene.add(planeEdge);

// ---------------------------------------------------------------------------
// 6. Timestamp label
// ---------------------------------------------------------------------------
// An HTML element that does not take part in the rotation. In 'rise' mode it
// sits at a fixed distance from the left edge and its height follows the
// centre of the image plane. That centre lies on the axis the camera orbits
// around, so its screen height changes only with time, never with rotation.
const stamp = document.getElementById("timestamp");
const anchor = new THREE.Vector3();

function placeStamp(y, label) {
  stamp.style.left = `${CONFIG.timestamp.left}px`;
  if (CONFIG.timestamp.mode === "fixed") {
    stamp.style.top = `${CONFIG.timestamp.top}px`;
  } else {
    anchor.set(0, y, 0).project(camera);
    stamp.style.top = `${((1 - anchor.y) / 2) * view.clientHeight}px`;
  }
  stamp.textContent = label;
}

// ---------------------------------------------------------------------------
// 7. Elevation view (lower panel)
// ---------------------------------------------------------------------------
// The lower view is an elevation of the finished sculpture: for each time
// slice, the full west-to-east extent of the storm as seen from the current
// angle, combining every depth. It turns with the camera, so it always shows
// the finished form's profile from the viewpoint above, and it does not change
// with playback. Rows stack upward through time.
const sectionCanvas = document.getElementById("section");
const ctx = sectionCanvas.getContext("2d");
let sectionAngle = null;
let sectionRows = [];
let sectionExtent = [-1, 1]; // where the frame itself begins and ends
const sRight = new THREE.Vector2(1, 0); // screen-right direction in (x, z)

// Contours converted once to scene coordinates (x, z).
const worldContours = frames.map((f) =>
  f.contours.map((c) => c.map(([u, v]) => [toX(u), toZ(v)])),
);

// Each connected piece of cloud projects onto the screen-right axis as one
// interval; overlapping intervals are merged. Holes lie inside their piece's
// interval, so they need no special treatment.
function projectedSpans(contours) {
  const spans = contours
    .map((c) => {
      let lo = Infinity,
        hi = -Infinity;
      for (const [x, z] of c) {
        const sVal = x * sRight.x + z * sRight.y;
        if (sVal < lo) lo = sVal;
        if (sVal > hi) hi = sVal;
      }
      return [lo, hi];
    })
    .sort((p, q) => p[0] - q[0]);
  const merged = [];
  for (const sp of spans) {
    const last = merged[merged.length - 1];
    if (last && sp[0] <= last[1]) last[1] = Math.max(last[1], sp[1]);
    else merged.push([...sp]);
  }
  return merged;
}

function frameExtent() {
  const xs = [-W / 2, W / 2],
    zs = [-D / 2, D / 2];
  const vals = xs.flatMap((x) => zs.map((z) => x * sRight.x + z * sRight.y));
  return [Math.min(...vals), Math.max(...vals)];
}

function computeSection(force = false) {
  const off = camera.position.clone().sub(controls.target);
  const a = Math.atan2(off.x, off.z);
  if (!force && sectionAngle !== null && Math.abs(a - sectionAngle) < 0.002)
    return;
  sectionAngle = a;
  sRight.set(Math.cos(a), -Math.sin(a));
  sectionRows = worldContours.map(projectedSpans);
  sectionExtent = frameExtent();
  drawSection();
}

let currentIndex = 0;
const hex = (n) => `#${n.toString(16).padStart(6, "0")}`;
function drawSection() {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const cw = sectionCanvas.clientWidth,
    ch = sectionCanvas.clientHeight;
  sectionCanvas.width = cw * dpr;
  sectionCanvas.height = ch * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  if (!sectionRows.length) return;

  // Keep the section's width proportional to the frame, centred.
  const w = Math.min(cw * CONFIG.section.width, ch * 1.6);
  const x0 = (cw - w) / 2;
  const span = ch * CONFIG.section.height;
  const top = (ch - span) / 2,
    bottom = top + span;
  const step = (bottom - top) / Math.max(N - 1, 1);
  ctx.lineWidth = Math.max(1, Math.min(2, step * 0.45));

  // Along-cut distance runs from -R to R (the volume's half-diagonal), so the
  // drawing keeps a constant scale as the cut turns.
  const R = Math.hypot(W, D) / 2;
  const sx = (sVal) => x0 + ((sVal + R) / (2 * R)) * w;

  // Faint guides marking the extent of the satellite frame from this angle.
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;
  for (const e of sectionExtent) {
    ctx.beginPath();
    ctx.moveTo(sx(e), top);
    ctx.lineTo(sx(e), bottom);
    ctx.stroke();
  }

  ctx.lineWidth = Math.max(1, Math.min(2, step * 0.45));
  sectionRows.forEach((spans, i) => {
    const y = bottom - i * step; // earliest at the bottom
    // Same states as the contour stack: frames already passed are orange like
    // the filled layers, the current frame is white like the image plane's
    // edge, and frames still to come are dim grey like the future outlines.
    ctx.strokeStyle = !CONFIG.section.lightUp
      ? "rgba(232,232,232,0.9)"
      : i === currentIndex
        ? "#ffffff"
        : i < currentIndex
          ? hex(CONFIG.colors.pastFill)
          : "rgba(143,143,143,0.4)";
    for (const [a, b] of spans) {
      ctx.beginPath();
      ctx.moveTo(sx(a), y);
      ctx.lineTo(sx(b), y);
      ctx.stroke();
    }
  });

  // The playhead, as a vertical rule beside the section; its orange mark is
  // the only part of the lower view that follows playback.
  ctx.strokeStyle = "rgba(232,232,232,0.6)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0 + w + 8, top);
  ctx.lineTo(x0 + w + 8, bottom);
  ctx.stroke();
  ctx.fillStyle = "#d8692e";
  ctx.fillRect(x0 + w + 5, bottom - currentIndex * step - 1, 7, 2);
}

// ---------------------------------------------------------------------------
// 8. Timeline, controls and the render loop
// ---------------------------------------------------------------------------
const playBtn = document.getElementById("play");
const timeInput = document.getElementById("time");

let t = 0; // current position in frames (fractional)
let playing = true;
let hold = 0;

function setPlaying(p) {
  playing = p;
  playBtn.textContent = p ? "Pause" : "Play";
  playBtn.setAttribute("aria-label", p ? "Pause" : "Play");
}

playBtn.addEventListener("click", () => setPlaying(!playing));
window.addEventListener("keydown", (e) => {
  if (e.code === "Space") {
    e.preventDefault();
    setPlaying(!playing);
  }
  // Press C after dragging to a view you like: the console prints the
  // elevation, azimuth and zoom to copy into CONFIG.camera.
  if (e.code === "KeyC") {
    const off = camera.position.clone().sub(controls.target);
    const elev = THREE.MathUtils.radToDeg(Math.asin(off.y / off.length()));
    const azim = THREE.MathUtils.radToDeg(Math.atan2(off.x, off.z));
    console.log(
      `camera: { elevation: ${elev.toFixed(1)}, azimuth: ${azim.toFixed(1)}, ` +
        `zoom: ${camera.zoom.toFixed(2)}, margin: ${CONFIG.camera.margin} }`,
    );
  }
});
timeInput.addEventListener("input", () => {
  setPlaying(false);
  t = Number(timeInput.value) * (N - 1);
});

function applyTime() {
  const y = (t / (N - 1)) * H;
  const i = Math.round(t);
  plane.position.y = y;
  planeEdge.position.y = y;
  plane.material.map = textures[i];
  setLayerStates(t);
  placeStamp(y, frames[i].label);
  if (i !== currentIndex) {
    currentIndex = i;
    drawSection();
  }
  timeInput.value = String(t / (N - 1));
}

const clock = new THREE.Clock();
function tick() {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (playing) {
    if (t >= N - 1) {
      hold += dt;
      if (hold > CONFIG.holdAtEnd) {
        t = 0;
        hold = 0;
      }
    } else {
      t = Math.min(N - 1, t + dt / CONFIG.secondsPerFrame);
    }
  }
  controls.update();
  computeSection(); // follows the camera's rotation
  applyTime();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

computeSection(true);
resize();
tick();

// Expose a small hook for testing and for screenshots from the console.
window.stormViewer = {
  setTime(frac) {
    setPlaying(false);
    t = frac * (N - 1);
    applyTime();
  },
};
