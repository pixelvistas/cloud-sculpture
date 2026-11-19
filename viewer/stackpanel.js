// Stack panel
// The acrylic sculpture as it will be built (stacks/<name>/stack.stl, written
// by slice_layers.py --viewer), drawn beside the contour view. It turns with
// the main camera, and each layer's edges change from white to green as the
// image plane passes the frame that layer was cut from.

import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

export async function createStackPanel(container, opts) {
  const { name, frames, ahead, passed, margin = 1.15 } = opts;

  // Which stack: the one named in CONFIG, or the first one exported
  let pick = name;
  if (!pick) {
    const index = await fetch('stacks/index.json').then((r) => (r.ok ? r.json() : null));
    pick = index?.stacks?.[0];
  }
  if (!pick) throw new Error('No stack exported yet (slice_layers.py --viewer viewer)');
  const info = await fetch(`stacks/${pick}/info.json`).then((r) => r.json());
  const buf = await fetch(`stacks/${pick}/stack.stl`).then((r) => {
    if (!r.ok) throw new Error(`Could not load stacks/${pick}/stack.stl (${r.status})`);
    return r.arrayBuffer();
  });
  const source = new STLLoader().parse(buf);

  // Split the triangles by layer. Walls sit inside a layer's height; a top
  // face (normal up) belongs to the layer below its height, a bottom face
  // (normal down) to the layer above it.
  const th = info.thickness, nLayers = info.layers;
  const P = source.getAttribute('position').array;
  const buckets = Array.from({ length: nLayers }, () => []);
  for (let i = 0; i < P.length; i += 9) {
    const z0 = P[i + 2], z1 = P[i + 5], z2 = P[i + 8];
    const cz = (z0 + z1 + z2) / 3;
    const flat = Math.max(z0, z1, z2) - Math.min(z0, z1, z2) < 1e-4;
    let k;
    if (!flat) k = Math.floor(cz / th);
    else {
      const ax = P[i + 3] - P[i], ay = P[i + 4] - P[i + 1];
      const bx = P[i + 6] - P[i], by = P[i + 7] - P[i + 1];
      const up = ax * by - ay * bx > 0;
      k = Math.round(cz / th) - (up ? 1 : 0);
    }
    k = Math.min(nLayers - 1, Math.max(0, k));
    for (let j = 0; j < 9; j++) buckets[k].push(P[i + j]);
  }

  // Scene: x = east, y = up, z = south, centred on the stack
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.environment = new THREE.PMREMGenerator(renderer)
    .fromScene(new RoomEnvironment(), 0.04).texture;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x303030, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(-120, 300, 180);
  scene.add(sun);

  const body = new THREE.MeshPhysicalMaterial({
    color: 0xeaf7f0, metalness: 0, roughness: 0.05, transmission: 1, thickness: 5,
    ior: 1.49, attenuationColor: new THREE.Color(passed), attenuationDistance: 40,
    flatShading: true, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
  });
  const group = new THREE.Group();
  const layers = buckets.map((arr) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3));
    g.rotateX(-Math.PI / 2);                     // (x, y, z) -> (x, z, -y): z south
    g.computeVertexNormals();
    const edgeMat = new THREE.LineBasicMaterial({ color: ahead, transparent: true, opacity: 0.85 });
    group.add(new THREE.Mesh(g, body));
    group.add(new THREE.LineSegments(new THREE.EdgesGeometry(g, 25), edgeMat));
    return { edgeMat };
  });
  const box = new THREE.Box3().setFromObject(group);
  const center = box.getCenter(new THREE.Vector3());
  group.position.sub(center);
  scene.add(group);
  const size = box.getSize(new THREE.Vector3());
  const R = Math.hypot(size.x, size.z) / 2, Hs = size.y;

  // The frame each layer was cut from: info.first is the first layer's
  // label, and layers follow every info.every frames.
  const first = Math.max(0, frames.findIndex((f) => f.label === info.first));
  const every = info.every || 1;
  const layerFrame = layers.map((_, k) => first + k * every);

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -2000, 2000);
  const aheadColor = new THREE.Color(ahead), passedColor = new THREE.Color(passed);

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    renderer.setSize(w, h, false);
  }
  window.addEventListener('resize', resize);
  resize();

  // Called every frame with the main camera, its target and the playback
  // position t (in frames, fractional) and the colour fade width.
  function update(mainCamera, target, t, fade = 1) {
    const dir = mainCamera.position.clone().sub(target).normalize();
    camera.position.copy(dir.multiplyScalar(500));
    camera.up.copy(mainCamera.up);
    camera.lookAt(0, 0, 0);
    const w = container.clientWidth, h = Math.max(container.clientHeight, 1);
    const aspect = w / h;
    const e = Math.asin(Math.min(1, Math.abs(dir.y)));
    const halfV = (Hs / 2) * Math.cos(e) + R * Math.sin(e);
    const fh = 2 * Math.max(halfV, R / aspect) * margin;
    camera.left = (-fh * aspect) / 2; camera.right = (fh * aspect) / 2;
    camera.top = fh / 2; camera.bottom = -fh / 2;
    camera.updateProjectionMatrix();
    const f = Math.max(fade, 1e-3);
    layers.forEach((L, k) => {
      const m = THREE.MathUtils.clamp((t - layerFrame[k]) / f + 0.5, 0, 1);
      L.edgeMat.color.lerpColors(aheadColor, passedColor, m);
    });
    renderer.render(scene, camera);
  }

  return { info, update };
}
