// The world: a z-up three.js scene whose camera orbits the spider horizontally. Everything
// drawn comes from the server's latest moment: the torso's place and quaternion (it tilts, and
// it can fall) and the world points of every leg's hip, knee, ankle and foot, as MuJoCo computed them.
// The page does no kinematics and no physics; the body's shapes are the model's (/api/describe),
// the eyes and chelicerae are the learning spider's dress.

import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/three/jsm/controls/OrbitControls.js';
import { BrainView } from '/static/vendor/brain_view.js';

// --- scene ---------------------------------------------------------------------------------
const canvas = document.getElementById('world');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1116);
scene.fog = new THREE.Fog(0x0e1116, 4, 14);

const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 50);
camera.up.set(0, 0, 1);                               // z is up, before the controls read it
camera.position.set(-1.2, -1.0, 0.75);
const controls = new OrbitControls(camera, canvas);
controls.enablePan = false;
controls.enableDamping = true;
controls.minPolarAngle = 1.15;                        // horizontal rotation only
controls.maxPolarAngle = 1.15;
controls.minDistance = 0.6;
controls.maxDistance = 24;                            // far enough to see the whole arena, its rim 10 m from the centre
// The fog begins beyond the spider at every distance, so zooming out shows the arena and not a haze.
function fogFor(r) { scene.fog.near = r + 3; scene.fog.far = r + 11; }
fogFor(camera.position.length());
// The wheel zooms by the distance scrolled, in the page's own handler: OrbitControls r160
// divides the wheel's delta by the device pixel ratio truncated to an integer, which is zero
// under a browser zoom below 100% (and on any display reporting a ratio under 1), so that one
// notch jumped to the nearest or the farthest distance and the zoom had two steps.
controls.enableZoom = false;
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const px = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 288 : 1);   // lines and pages as pixels
  const k = Math.exp(px * 0.0015);                      // a 100 px notch: 16% nearer or farther
  const offset = camera.position.clone().sub(controls.target);
  const r = Math.min(controls.maxDistance, Math.max(controls.minDistance, offset.length() * k));
  camera.position.copy(controls.target).add(offset.setLength(r));
  fogFor(r);
}, { passive: false });

const sky = new THREE.HemisphereLight(0xd9e6ff, 0x202830, 0.9);
scene.add(sky);
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
const LIGHT = { sky: sky.intensity, sun: sun.intensity };   // the page's own; the world panel scales both
sun.position.set(2, -3, 5);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = sun.shadow.camera.bottom = -3;
sun.shadow.camera.right = sun.shadow.camera.top = 3;
scene.add(sun);
scene.add(sun.target);

const GROUND_COLOUR = '#1b2230';                      // the page's own; the view panel changes it, for this browser only
// The ground's height, the body's ground_height: 0 on the flat arena (MuJoCo's plane), the sum of
// the server's waves (wavelength, direction, amplitude, phase) on the hills and the steep hills,
// read at the drawing's point plus the pose's offset (a recentre moves the ground with the body).
// The page draws the server's terrain; it never has one of its own.
let waves = [], offset = [0, 0];
let plateaus = [];                                     // the rocks arena's raised ground, in the terrain's frame
let rockHills = [];                                    // the rocks arena's rolling floor: body.py's ROCK_HILLS
function setArena(name, terrains, at) {
  waves = ((terrains || {})[name] || []).map(([w, d, a, ph]) => [2 * Math.PI / w * Math.cos(d), 2 * Math.PI / w * Math.sin(d), a, ph]);
  offset = at || [0, 0];
  const law = name === 'rocks' && described && described.rocks, reach = GROUND_RADIUS;
  plateaus = [];
  rockHills = law ? law.hills.map(([w, d, a, ph]) => [2 * Math.PI / w, d, a, ph]) : [];
  if (law)
    for (let i = Math.floor((offset[0] - reach - law.plateau_shift[0]) / law.plateau_cell) - 1; i <= Math.floor((offset[0] + reach - law.plateau_shift[0]) / law.plateau_cell) + 1; i++)
      for (let j = Math.floor((offset[1] - reach - law.plateau_shift[1]) / law.plateau_cell) - 1; j <= Math.floor((offset[1] + reach - law.plateau_shift[1]) / law.plateau_cell) + 1; j++) {
        const r = rockOf(i, j, law, PLATEAU);
        if (r.height) plateaus.push(r);
      }
}
function plateauHeight(X, Y) {                         // body.py's plateau layer at terrain point X, Y (0 off them)
  let h = 0;
  for (const r of plateaus) h = Math.max(h, rockHeight(r, X, Y));
  return h;
}
function hillsHeight(X, Y) {                          // body.py's _rock_hills at terrain point X, Y
  let h = 0;
  for (const [k, d, a, ph] of rockHills) h += a * (1 + Math.cos(k * Math.cos(d) * X + k * Math.sin(d) * Y + ph));
  return h;
}
function floorHeight(X, Y) {                           // the rocks' floor: the higher of the hills and a plateau
  return Math.max(hillsHeight(X, Y), plateauHeight(X, Y));
}
function groundHeight(x, y) {
  let h = plateaus.length || rockHills.length ? floorHeight(x + offset[0], y + offset[1]) : 0;
  for (const [kx, ky, a, ph] of waves) h += a * Math.cos(kx * (x + offset[0]) + ky * (y + offset[1]) + ph);
  return h;
}
function lay(geometry, lift) {                        // lifts every vertex of a z-up geometry onto the ground
  const p = geometry.attributes.position;
  for (let i = 0; i < p.count; i++) p.setZ(i, groundHeight(p.getX(i), p.getY(i)) + lift);
  p.needsUpdate = true;
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
}
const ARENA_RADIUS = 10;                              // body.py's ARENA_RADIUS: the recentre's rim
const GROUND_RADIUS = ARENA_RADIUS + 0.5;
const ground = new THREE.Mesh(new THREE.RingGeometry(0.001, GROUND_RADIUS, 320, 220), new THREE.MeshStandardMaterial({ color: GROUND_COLOUR, roughness: 1 }));
ground.receiveShadow = true;
scene.add(ground);
const GRID_COLOUR = '#262e3c';                        // the page's own; the two axes through the origin a shade lighter
let grid = null, gridHex = GRID_COLOUR;
function makeGrid(hex) {                              // the lines follow the ground: built from short segments, rebuilt to recolour
  gridHex = hex;
  if (grid) { scene.remove(grid); grid.geometry.dispose(); grid.material.dispose(); }
  const line = new THREE.Color(hex), axes = new THREE.Color(hex).lerp(new THREE.Color('#ffffff'), 0.12);
  const pos = [], col = [], STEP = waves.length || plateaus.length || rockHills.length ? 0.05 : 11;   // flat: each line one segment
  for (let i = -Math.floor(GROUND_RADIUS / 0.25); i <= Math.floor(GROUND_RADIUS / 0.25); i++) {
    const c = i === 0 ? axes : line, u = i * 0.25, end = Math.sqrt(Math.max(0, GROUND_RADIUS * GROUND_RADIUS - u * u));   // the grid ends with the ground disc
    for (let t = -end; t < end - 1e-9; t += STEP) {
      const t1 = Math.min(t + STEP, end);
      for (const [x0, y0, x1, y1] of [[t, u, t1, u], [u, t, u, t1]]) {
        pos.push(x0, y0, groundHeight(x0, y0) + 0.001, x1, y1, groundHeight(x1, y1) + 0.001);
        col.push(c.r, c.g, c.b, c.r, c.g, c.b);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  grid = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true }));
  scene.add(grid);
}
// The rocks: spider/body.py's rocks, cell by cell, from the server's law (describe's rocks) and the
// rough ground's hash, so the page draws the rocks the body meets: each rock the field MuJoCo holds,
// sampled on the same lattice (the law's step) and flat shaded, drawn around the pose's offset.
const M64 = (1n << 64n) - 1n;
function grain(i, j) {                                // body.py's _grain: the same 64-bit hash, in -1..1
  let h = ((BigInt.asUintN(64, BigInt(i)) * 0x9E3779B97F4A7C15n) ^ (BigInt.asUintN(64, BigInt(j)) * 0xC2B2AE3D27D4EB4Fn)) & M64;
  h ^= h >> 29n;
  h = (h * 0xBF58476D1CE4E5B9n) & M64;
  h ^= h >> 32n;
  return Number(h >> 11n) / 9007199254740992 * 2 - 1;
}
const GIANT_ROW = 2 ** 40;                            // body.py's GIANT_ROW: a lattice's draws lie this far along j times its number
const SMALL = 0, GIANT = 1, PLATEAU = 2;              // body.py's lattices
const plateauCache = new Map();
function offPlateaus(cx, cy, radius, law, giant) {    // body.py's _off_plateaus: off every bare plateau's foot (a giant off every one)
  const pi = Math.floor((cx - law.plateau_shift[0]) / law.plateau_cell), pj = Math.floor((cy - law.plateau_shift[1]) / law.plateau_cell);
  for (let di = -1; di <= 1; di++)
    for (let dj = -1; dj <= 1; dj++) {
      const key = (pi + di) + ',' + (pj + dj);
      if (!plateauCache.has(key)) plateauCache.set(key, rockOf(pi + di, pj + dj, law, PLATEAU));
      const p = plateauCache.get(key);
      if (p.height && !(p.rocky && !giant) && Math.hypot(cx - p.cx, cy - p.cy) < p.radius + radius) return false;
    }
  return true;
}
function rockOf(i, j, law, layer = SMALL) {           // body.py's _rock_shape: 0 height for none
  const u = Array.from({ length: 48 }, (_, k) => (grain(48 * i + k, j + layer * GIANT_ROW) + 1) / 2), F = law.facets;
  const big = u[10] < law.big, giant = layer !== SMALL;
  const [cell, share, hr, rr] = layer === PLATEAU ? [law.plateau_cell, law.plateau_share, law.plateau_height, [0, 0]]
    : layer === GIANT ? [law.giant_cell, law.giant_share, law.giant_height, law.giant_radius]
    : [law.cell, law.share, big ? law.big_height : law.height, big ? law.big_radius : law.radius];
  const height = hr[0] + u[1] * (hr[1] - hr[0]), plateau = layer === PLATEAU;
  const narrow = plateau ? law.plateau_stretch + u[45] * (1 - law.plateau_stretch) : law.stretch + u[45] * (1 - law.stretch);
  const ramp = law.plateau_slope[0] + u[6] * (law.plateau_slope[1] - law.plateau_slope[0]);   // a plateau's: its top's reach chosen, its ramps added
  const level = Array.from({ length: F }, (_, k) => (law.plateau_top[0] + u[13 + k] * (law.plateau_top[1] - law.plateau_top[0])) / narrow + height / ramp);
  const radius = plateau ? 1.25 * Math.max(...level) : rr[0] + u[2] * (rr[1] - rr[0]);
  const [sx, sy] = plateau ? law.plateau_shift : [0, 0];   // body.py's _shift
  const cx = sx + i * cell + radius + u[3] * (cell - 2 * radius), cy = sy + j * cell + radius + u[4] * (cell - 2 * radius);
  const flanks = Array.from({ length: F }, (_, k) => {
    const angle = 2 * Math.PI * u[5] + 2 * Math.PI * (k + law.jitter * (u[21 + k] - 0.5)) / F;
    return [Math.cos(angle), Math.sin(angle), plateau ? level[k] : radius * (law.foot[0] + u[13 + k] * (law.foot[1] - law.foot[0])), angle + Math.PI / F];
  });
  const reach = law.apex * radius * u[11], turn = 2 * Math.PI * u[12];
  const there = u[0] < share && Math.hypot(cx, cy) - (giant ? radius : 0) >= law.clear && (plateau || offPlateaus(cx, cy, radius, law, layer === GIANT));
  if (plateau)                                        // level-topped, ramped all round, never a spike
    return { cx, cy, radius, flanks, spike: false, rocky: u[9] < law.plateau_rocky, height: there ? height : 0, slope: ramp,
             crown: flanks.map(f => [Math.cos(f[3]), Math.sin(f[3]), 0, height]),
             lean: 2 * Math.PI * u[7], tilt: 0, apex: [0, 0],
             axis: 2 * Math.PI * u[44], narrow, tone: u[46], warm: u[47] };
  const spike = u[9] < law.spike;
  return { cx, cy, radius, flanks, spike, height: there ? height : 0,
           slope: spike ? 1.05 * height / Math.min(...flanks.map(f => f[2])) : law.slope[0] + u[6] * (law.slope[1] - law.slope[0]),
           crown: flanks.map((f, k) => [Math.cos(f[3]), Math.sin(f[3]), law.crown[0] + u[29 + k] * (law.crown[1] - law.crown[0]), height * (0.85 + 0.15 * u[37 + k])]),
           lean: 2 * Math.PI * u[7], tilt: spike ? 0 : law.tilt * u[8],
           apex: spike ? [reach * Math.cos(turn), reach * Math.sin(turn)] : [0, 0],
           axis: 2 * Math.PI * u[44], narrow, tone: u[46], warm: u[47] };                // the page's own: the rock's colour, not the world's
}
function rockHeight(r, x, y) {                        // body.py's _rocks_height for one rock
  const dx = x - r.cx, dy = y - r.cy, ax = Math.cos(r.axis), ay = Math.sin(r.axis);
  const px = ax * dx + ay * dy - r.apex[0], py = (-ay * dx + ax * dy) / r.narrow - r.apex[1];
  if (!r.height || Math.hypot(dx, dy) >= r.radius) return 0;
  let h = r.height * (1 - r.tilt * (1 + (Math.cos(r.lean) * dx + Math.sin(r.lean) * dy) / r.radius) / 2);
  for (const [nx, ny, foot] of r.flanks) h = Math.min(h, r.slope * (foot - (nx * px + ny * py)));
  for (const [ux, uy, crown, rise] of r.crown) h = Math.min(h, rise - (r.spike ? r.slope : crown) * (ux * px + uy * py));
  return Math.max(h, 0);
}
const rockSkin = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0.0, flatShading: true, side: THREE.DoubleSide });
const rocks = new THREE.Group();
scene.add(rocks);
function rockMesh(r, law, ox, oy, s, lift = 0) {      // the field's triangles over the rock (every s m), its bare ground left out, raised by lift
  const i0 = Math.floor((r.cx - r.radius) / s), i1 = Math.ceil((r.cx + r.radius) / s);
  const j0 = Math.floor((r.cy - r.radius) / s), j1 = Math.ceil((r.cy + r.radius) / s), n = i1 - i0 + 1;
  const position = [], z = [], colour = [], inside = [];
  const base = new THREE.Color().setHSL(r.warm < 0.5 ? 0.6 : 0.08, 0.01 + 0.07 * Math.abs(2 * r.warm - 1), 0.2 + 0.42 * r.tone), c = new THREE.Color();   // basalt to limestone, cool or warm grey
  for (let j = j0; j <= j1; j++)
    for (let i = i0; i <= i1; i++) {
      const hills = hillsHeight(i * s, j * s), speck = (grain(i, j) + 1) / 2;   // a fleck per node, darker at the foot
      const h = rockHeight(r, i * s, j * s), under = r.plateau ? lift : floorHeight(i * s, j * s);
      z.push(r.plateau ? Math.max(h - hills, 0) : h);   // a plateau of rock shows where it rises out of the hills
      inside.push(Math.hypot(i * s - ox, j * s - oy) <= GROUND_RADIUS);   // cut at the ground's disc, as the ground is
      position.push(i * s - ox, j * s - oy, under + (r.plateau ? Math.max(h, hills) : h));   // a small rock on the hill or plateau it lies on
      c.copy(base).multiplyScalar(0.72 + 0.28 * Math.min(1, h / Math.max(r.height, 1e-9) * 1.6) + 0.12 * (speck - 0.5));
      colour.push(c.r, c.g, c.b);
    }
  const index = [], at = (i, j) => (j - j0) * n + (i - i0);
  for (let j = j0; j < j1; j++)
    for (let i = i0; i < i1; i++) {
      const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
      if ((z[a] || z[b] || z[c]) && inside[a] && inside[b] && inside[c]) index.push(a, b, c);
      if ((z[a] || z[c] || z[d]) && inside[a] && inside[c] && inside[d]) index.push(a, c, d);
    }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(position, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(colour, 3));
  g.setIndex(index);
  g.computeVertexNormals();
  const m = new THREE.Mesh(g, rockSkin);
  m.castShadow = m.receiveShadow = true;
  return m;
}
function drawRocks(law, at) {                         // the small rocks on the field's lattice, the giants on every second node of it, the plateaus of rock every 10 cm
  for (const m of rocks.children) m.geometry.dispose();
  rocks.clear();
  if (!law) return;
  const [ox, oy] = at || [0, 0], reach = GROUND_RADIUS;
  for (const p of plateaus)                           // a plateau of rock: drawn as rock, a little above the ground it covers
    if (p.rocky && Math.hypot(p.cx - ox, p.cy - oy) <= reach + p.radius) rocks.add(rockMesh({ ...p, plateau: true }, law, ox, oy, 0.1, 0.004));
  for (const [layer, cell, step] of [[SMALL, law.cell, law.step], [GIANT, law.giant_cell, 2 * law.step]])
    for (let i = Math.floor((ox - reach) / cell) - 1; i <= Math.floor((ox + reach) / cell) + 1; i++)
      for (let j = Math.floor((oy - reach) / cell) - 1; j <= Math.floor((oy + reach) / cell) + 1; j++) {
        const r = rockOf(i, j, law, layer);
        if (r.height && Math.hypot(r.cx - ox, r.cy - oy) <= reach + r.radius) rocks.add(rockMesh(r, law, ox, oy, step));
      }
}
const rim = new THREE.Mesh(new THREE.RingGeometry(ARENA_RADIUS - 0.03, ARENA_RADIUS + 0.03, 512), new THREE.MeshBasicMaterial({ color: 0x357a38, side: THREE.DoubleSide }));
scene.add(rim);
let arenaDrawn = null;
function drawArena(name, terrains, at) {               // the ground, its grid and the rim, laid on the arena's terrain
  const key = name + ' ' + (name === 'flat' ? '' : (at || [0, 0]).join(','));
  if (key === arenaDrawn || (name !== 'flat' && !terrains)) return;   // a terrain is drawn once its waves are described
  arenaDrawn = key;
  setArena(name, terrains, at);
  drawRocks(name === 'rocks' ? described && described.rocks : null, at);
  lay(ground.geometry, 0);
  lay(rim.geometry, 0.002);
  makeGrid(gridHex);
}
drawArena('flat');

// --- the surface: a meadow, gravel, or both, painted on the ground. How the world is drawn, not
// its law: the body's contact and the reward are the same on every surface, the brain senses no
// colour, and the choice lives in this browser only. Drawn on canvases here, nothing is loaded.
const SURFACE_TILE = 0.5;                             // metres one tile covers
function seeded(seed) {                               // a small deterministic generator, so a surface looks the same on every load
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function tileCanvas(size) { const c = document.createElement('canvas'); c.width = c.height = size; return c; }
function wrapped(size, x, y, r, draw) {               // draws at a point and at its copies across the edges, so the tile repeats seamlessly
  for (const dx of [-size, 0, size]) for (const dy of [-size, 0, size]) {
    if (x + dx < -r || x + dx > size + r || y + dy < -r || y + dy > size + r) continue;
    draw(x + dx, y + dy);
  }
}
function meadowTile(size = 512) {
  const colour = tileCanvas(size), height = tileCanvas(size), rnd = seeded(7);
  const c = colour.getContext('2d'), h = height.getContext('2d');
  c.fillStyle = '#2f4a1f'; c.fillRect(0, 0, size, size);
  h.fillStyle = '#404040'; h.fillRect(0, 0, size, size);
  for (let i = 0; i < 260; i++) {                     // soft patches of lighter and darker soil under the grass
    const x = rnd() * size, y = rnd() * size, r = 10 + rnd() * 40, g = rnd() < 0.5 ? 'rgba(70,95,35,0.25)' : 'rgba(25,40,15,0.3)';
    wrapped(size, x, y, r, (px, py) => { c.fillStyle = g; c.beginPath(); c.arc(px, py, r, 0, 2 * Math.PI); c.fill(); });
  }
  c.lineCap = h.lineCap = 'round';
  for (let i = 0; i < 9000; i++) {                    // blades: short strokes leaning one way, lighter at the tip
    const x = rnd() * size, y = rnd() * size, len = 4 + rnd() * 10, lean = -0.6 + rnd() * 1.2 - 1.57;
    const ex = Math.cos(lean) * len, ey = Math.sin(lean) * len;
    const l = 22 + rnd() * 22, hue = 78 + rnd() * 34, w = 0.8 + rnd() * 1.1;
    wrapped(size, x, y, len, (px, py) => {
      c.strokeStyle = `hsl(${hue},${45 + rnd() * 20}%,${l}%)`; c.lineWidth = w;
      c.beginPath(); c.moveTo(px, py); c.lineTo(px + ex, py + ey); c.stroke();
      const v = 90 + Math.round(rnd() * 120); h.strokeStyle = `rgb(${v},${v},${v})`; h.lineWidth = w;
      h.beginPath(); h.moveTo(px, py); h.lineTo(px + ex, py + ey); h.stroke();
    });
  }
  for (let i = 0; i < 40; i++) {                      // a few small flowers, white and yellow
    const x = rnd() * size, y = rnd() * size, col = rnd() < 0.6 ? '#e8e4d0' : '#e2c43c';
    wrapped(size, x, y, 3, (px, py) => { c.fillStyle = col; c.beginPath(); c.arc(px, py, 1.6, 0, 2 * Math.PI); c.fill(); });
  }
  return { colour, height };
}
function gravelTile(size = 512) {
  const colour = tileCanvas(size), height = tileCanvas(size), rnd = seeded(11);
  const c = colour.getContext('2d'), h = height.getContext('2d');
  c.fillStyle = '#3a352e'; c.fillRect(0, 0, size, size);
  h.fillStyle = '#202020'; h.fillRect(0, 0, size, size);
  for (let i = 0; i < 2600; i++) {                    // stones: rounded, of greys and browns, lit from one side
    const x = rnd() * size, y = rnd() * size, rx = 3 + rnd() * 9, ry = rx * (0.55 + rnd() * 0.45), rot = rnd() * Math.PI;
    const base = 70 + rnd() * 90, warm = rnd() * 18;
    const fill = `rgb(${base + warm},${base + warm * 0.6},${base})`, lit = `rgba(255,250,240,${0.12 + rnd() * 0.12})`;
    const hv = 120 + Math.round(rnd() * 135);
    wrapped(size, x, y, rx + 2, (px, py) => {
      c.fillStyle = 'rgba(0,0,0,0.35)'; c.beginPath(); c.ellipse(px + 1.2, py + 1.6, rx, ry, rot, 0, 2 * Math.PI); c.fill();
      c.fillStyle = fill; c.beginPath(); c.ellipse(px, py, rx, ry, rot, 0, 2 * Math.PI); c.fill();
      c.fillStyle = lit; c.beginPath(); c.ellipse(px - rx * 0.25, py - ry * 0.3, rx * 0.5, ry * 0.45, rot, 0, 2 * Math.PI); c.fill();
      const g = h.createRadialGradient(px, py, 0, px, py, rx);
      g.addColorStop(0, `rgb(${hv},${hv},${hv})`); g.addColorStop(1, 'rgb(40,40,40)');
      h.fillStyle = g; h.beginPath(); h.ellipse(px, py, rx, ry, rot, 0, 2 * Math.PI); h.fill();
    });
  }
  return { colour, height };
}
// Meadow and gravel together: the whole disc on one canvas, gravel under a meadow cut by a
// smooth mask of the same waves wherever it is drawn, so paths of gravel wind through the grass.
function mixedDisc(meadow, gravel, per = 186) {
  const span = 11, size = per * (span / SURFACE_TILE);   // whole tiles, so the disc's canvas repeats seamlessly every 11 m
  const out = { colour: tileCanvas(size), height: tileCanvas(size) };
  const M = 256, mask = tileCanvas(M), m = mask.getContext('2d'), img = m.createImageData(M, M);
  for (let j = 0; j < M; j++) for (let i = 0; i < M; i++) {
    const u = 2 * Math.PI * i / M, w = 2 * Math.PI * j / M;   // whole waves over the 11 m, so the mask repeats with the canvas
    const v = Math.sin(2 * u + 0.4) + Math.sin(2 * w - 1.1) + 0.8 * Math.sin(u + w + 2.0) + 0.6 * Math.sin(4 * u - 3 * w);
    const a = Math.max(0, Math.min(1, (v + 0.35) / 0.5));   // meadow where the sum is high, a soft edge of a few centimetres
    img.data[(j * M + i) * 4 + 3] = Math.round(255 * a);
  }
  m.putImageData(img, 0, 0);
  for (const key of ['colour', 'height']) {
    const ctx = out[key].getContext('2d'), layer = tileCanvas(size), l = layer.getContext('2d');
    const paint = (target, src) => { const p = target.createPattern(src[key], 'repeat'); p.setTransform(new DOMMatrix().scale(per / src[key].width)); target.fillStyle = p; target.fillRect(0, 0, size, size); };
    paint(ctx, gravel);
    paint(l, meadow);
    l.globalCompositeOperation = 'destination-in'; l.imageSmoothingEnabled = true; l.drawImage(mask, 0, 0, size, size);
    ctx.drawImage(layer, 0, 0);
  }
  return out;
}
const surfaces = {};                                   // built on first use, then kept
function surfaceMaps(name) {
  if (surfaces[name]) return surfaces[name];
  const made = name === 'meadow' ? meadowTile() : name === 'gravel' ? gravelTile()
    : mixedDisc(surfaceMaps('meadow').source, surfaceMaps('gravel').source);
  const repeat = name === 'mixed' ? 1 : 11 / SURFACE_TILE;   // the ring's uv spans its 11 m diameter once
  const texture = (canvas, colour) => {
    const t = new THREE.CanvasTexture(canvas);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(repeat, repeat);
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
    if (colour) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  };
  return (surfaces[name] = { source: made, map: texture(made.colour, true), bump: texture(made.height, false) });
}
let surfaceName = 'plain', groundHex = GROUND_COLOUR;
function setSurface(name) {
  surfaceName = name;
  const mat = ground.material;
  if (name === 'plain') { mat.map = null; mat.bumpMap = null; mat.color.set(groundHex); }
  else {
    const s = surfaceMaps(name);
    mat.map = s.map; mat.bumpMap = s.bump; mat.bumpScale = name === 'meadow' ? 1.2 : 2.5;
    mat.color.set('#ffffff');
  }
  mat.needsUpdate = true;
}
let viewFollows = true;                             // the view panel's choice: the camera follows the spider, or stays
function setView(value) { viewFollows = value !== 'fixed'; }
function setGroundColour(hex) { groundHex = hex; if (surfaceName === 'plain') ground.material.color.set(hex); }

// --- the spider ----------------------------------------------------------------------------
// The learning spider's dress on the model's body: the torso ellipsoid is the prosoma that
// carries the legs, the abdomen and waist are the model's too; eyes and chelicerae are the page's
// (the model's head sphere is under the chelicerae). The group is posed by the torso's place and quaternion;
// a leg runs from inside the prosoma (the coxa) to MuJoCo's hip, knee, ankle and foot, brought into the
// torso's frame. The page does no kinematics: every joint point is the model's.
// Drawn radii, chosen by eye after a red-knee tarantula's photograph (a leg about a fifth of the
// prosoma's width, hair included); the model's collision capsules stay thinner (shape.leg_radius).
const FEMUR_R = 0.021, TIBIA_R = 0.018, COXA_R = 0.024, TARSUS_R = 0.013;
const ROOT_Y = 0.03;                                  // where a leg leaves the body: well inside the ellipsoid
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _h = new THREE.Vector3(), _r = new THREE.Vector3(), _d = new THREE.Vector3(), _k = new THREE.Vector3(), _yUp = new THREE.Vector3(0, 1, 0);
function placeCylinder(mesh, a, b, radius) {
  _d.subVectors(b, a);
  const length = _d.length();
  mesh.position.copy(a).addScaledVector(_d, 0.5);
  mesh.scale.set(radius, length, radius);
  mesh.quaternion.setFromUnitVectors(_yUp, _d.normalize());
}
const spider = new THREE.Group();
scene.add(spider);
// Colours after a red-knee tarantula's photograph: black legs with orange knees and pale rings,
// an orange-tan prosoma with a dark star at its centre, a dark brown abdomen with brown hair.
// Underneath, as on a tarantula's ventral side: a black sternum under the prosoma and a black
// belly under the abdomen with two pairs of pale book-lung patches at its front, so a body on
// its back shows its underside.
const BLACK = 0x15110e, ORANGE = 0xe0662a, RING = 0xd08850, TAN = 0xd48a4c, STAR = 0x2a1a12;
const ABDOMEN_BROWN = 0x24170f, STERNUM = 0x0b0807, BELLY = 0x0a0706, BOOK_LUNG = 0xb8a07a;
const legSkin = new THREE.MeshStandardMaterial({ color: BLACK, roughness: 0.85, metalness: 0.05 });
const bandSkin = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0.05 });   // painted legs
const proSkin = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8, metalness: 0.05 });     // painted prosoma
const abdomenSkin = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0.0 });   // painted abdomen
const kneeSkin = new THREE.MeshStandardMaterial({ color: ORANGE, roughness: 0.8, metalness: 0.0 });
const ankleSkin = new THREE.MeshStandardMaterial({ color: RING, roughness: 0.85, metalness: 0.0 });
const hairSkin = new THREE.LineBasicMaterial({ color: 0x0a0807, transparent: true, opacity: 0.8 });
const paintedHair = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85 });
const _c = new THREE.Color();
// Paint a geometry vertex by vertex: colour(x, y, z) in its own unit space gives a hex colour.
function paint(geometry, colour) {
  const p = geometry.attributes.position, cols = [];
  for (let i = 0; i < p.count; i++) { _c.setHex(colour(p.getX(i), p.getY(i), p.getZ(i))); cols.push(_c.r, _c.g, _c.b); }
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  return geometry;
}
// The prosoma: tan, with a dark star of eight rays at the top centre and the black sternum below.
const prosomaColour = (x, y, z) => (z < -0.3 ? STERNUM : z + 0.14 * Math.cos(8 * Math.atan2(y, x)) > 0.72 ? STAR : TAN);
// The abdomen in its own unit space (x toward the prosoma, z up): brown above, black below, and
// the book lungs as two pale pairs on the belly's front.
const bookLung = (x, y, z) => z < -0.45 && Math.abs(Math.abs(y) - 0.3) < 0.12 && (Math.abs(x - 0.62) < 0.08 || Math.abs(x - 0.38) < 0.08);
const abdomenColour = (x, y, z) => (bookLung(x, y, z) ? BOOK_LUNG : z < -0.25 ? BELLY : ABDOMEN_BROWN);
// Along a leg segment (y from -0.5 at its root to 0.5 at its end): the femur's end, the patella
// and a ring before the ankle are orange; the rest is black.
const femurColour = (x, y) => (y > 0.38 ? ORANGE : BLACK);
const tibiaColour = (x, y) => (y < -0.3 ? ORANGE : y > 0.2 && y < 0.32 ? RING : BLACK);
const tarsusColour = (x, y) => (y < -0.3 ? RING : BLACK);
let hairSeed = 7;
const hairRand = () => (hairSeed = (hairSeed * 1664525 + 1013904223) % 4294967296) / 4294967296;
// Bristles in a mesh's own unit space: they inherit its scale, so a scaled cylinder or sphere stays hairy.
// colour: a hex, a function (x, y, z) -> hex for painted hair, or none for black.
function hairs(count, onSphere, lengthMin, lengthMax, colour) {
  const pts = [], cols = [];
  for (let i = 0; i < count; i++) {
    const len = lengthMin + (lengthMax - lengthMin) * hairRand();
    if (onSphere) {
      const u = 2 * hairRand() - 1, t = 2 * Math.PI * hairRand(), r = Math.sqrt(1 - u * u);
      const x = r * Math.cos(t), y = u, z = r * Math.sin(t);
      pts.push(x, y, z, x * (1 + len), y * (1 + len), z * (1 + len));
      if (colour) { _c.setHex(typeof colour === 'function' ? colour(x, y, z) : colour); cols.push(_c.r, _c.g, _c.b, _c.r, _c.g, _c.b); }
    } else {
      const t = 2 * Math.PI * hairRand(), y = hairRand() - 0.5, tilt = 0.06 * (hairRand() - 0.5);
      const x = Math.cos(t), z = Math.sin(t);
      pts.push(x, y, z, x * (1 + len), y + tilt, z * (1 + len));
      if (colour) { _c.setHex(typeof colour === 'function' ? colour(x, y, z) : colour); cols.push(_c.r, _c.g, _c.b, _c.r, _c.g, _c.b); }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  if (!colour) return new THREE.LineSegments(g, hairSkin);
  g.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  return new THREE.LineSegments(g, paintedHair);
}
const prosoma = new THREE.Mesh(paint(new THREE.SphereGeometry(1, 48, 32), prosomaColour), proSkin);   // the model's torso ellipsoid
prosoma.castShadow = true;
prosoma.add(hairs(500, true, 0.08, 0.18, (x, y, z) => (z < -0.3 ? 0x1a1410 : prosomaColour(x, y, z) === STAR ? STAR : 0xf0b070)));
spider.add(prosoma);
const abdomen = new THREE.Mesh(paint(new THREE.SphereGeometry(1, 64, 48), abdomenColour), abdomenSkin);
abdomen.castShadow = true;
abdomen.add(hairs(1200, true, 0.08, 0.22, (x, y, z) => (z < -0.25 ? 0x1a1410 : 0x7a5034)));   // brown hair above, black below
spider.add(abdomen);
const waist = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 12), legSkin);   // the pedicel
spider.add(waist);
const eyeSkin = new THREE.MeshStandardMaterial({ color: 0x050505, roughness: 0.08, metalness: 0.3 });   // black and glossy
// Eight eyes on the prosoma's front, after a tarantula's face: two large ones in the middle, a row
// of four small ones below them, and one small eye on each side, higher and further out.
// Each is (y, z, radius) in metres; buildBody seats it on the torso ellipsoid.
const EYES = [
  [0.012, 0.040, 0.009], [-0.012, 0.040, 0.009],      // the large middle pair
  [0.006, 0.026, 0.005], [-0.006, 0.026, 0.005],      // the row of four below
  [0.019, 0.027, 0.0045], [-0.019, 0.027, 0.0045],
  [0.030, 0.044, 0.0045], [-0.030, 0.044, 0.0045],    // the side pair, higher and further out
];
const EYE_SINK = 0.4;                                  // share of an eye's radius set into the body
const eyes = EYES.map(([, , r]) => {
  const eye = new THREE.Mesh(new THREE.SphereGeometry(r, 12, 10), eyeSkin);
  spider.add(eye);
  return eye;
});
function seatEyes([a, b, c]) {
  // An eye's centre lies on the torso ellipsoid at its (y, z), moved inward along the surface
  // normal so the eye sits in the body instead of floating in front of it.
  EYES.forEach(([y, z, r], i) => {
    const x = a * Math.sqrt(Math.max(0, 1 - (y / b) ** 2 - (z / c) ** 2));
    const n = new THREE.Vector3(x / (a * a), y / (b * b), z / (c * c)).normalize();
    eyes[i].position.set(x, y, z).addScaledVector(n, -EYE_SINK * r);
  });
}
// Chelicerae: two stout basal segments reaching forward and down from under the eyes, each
// ending in a fang that curves down and inward, the way a tarantula's hang. Drawn in the frame of
// the model's head sphere (buildBody puts the group at head_x) and kept inside it (radius 0.055):
// the floor stops at that sphere, so a fang inside it never reaches into the ground.
const fangSkin = new THREE.MeshStandardMaterial({ color: 0x0b0d12, roughness: 0.3, metalness: 0.45 });
const chelicerae = new THREE.Group();
spider.add(chelicerae);
for (const sgn of [-1, 1]) {
  const root = new THREE.Vector3(-0.015, sgn * 0.022, -0.010), joint = new THREE.Vector3(0.020, sgn * 0.024, -0.028);
  const base = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 12), legSkin);
  placeCylinder(base, root, joint, 0.013);
  base.castShadow = true;
  base.add(hairs(40, false, 0.3, 0.6));
  chelicerae.add(base);
  const knuckle = new THREE.Mesh(new THREE.SphereGeometry(0.013, 10, 8), legSkin);   // rounds the joint
  knuckle.position.copy(joint);
  chelicerae.add(knuckle);
  const curve = new THREE.QuadraticBezierCurve3(joint, new THREE.Vector3(0.032, sgn * 0.020, -0.036),
                                                new THREE.Vector3(0.027, sgn * 0.013, -0.042));
  const fang = new THREE.Mesh(new THREE.TubeGeometry(curve, 10, 0.0045, 8, false), fangSkin);
  fang.castShadow = true;
  chelicerae.add(fang);
  const tip = new THREE.Mesh(new THREE.ConeGeometry(0.0045, 0.010, 8), fangSkin);   // the point, along the curve's end
  const end = curve.getPoint(1), tangent = curve.getTangent(1);
  tip.position.copy(end).addScaledVector(tangent, 0.004);
  tip.quaternion.setFromUnitVectors(_yUp, tangent);
  chelicerae.add(tip);
}
const unit = new THREE.CylinderGeometry(1, 1, 1, 12);
const femurShape = paint(new THREE.CylinderGeometry(1, 0.85, 1, 12, 16), femurColour);   // +y end goes to the knee
const tibiaShape = paint(new THREE.CylinderGeometry(0.8, 1, 1, 12, 16), tibiaColour);   // knee (-y) to ankle (+y), tapering a little
let tarsusShape = null;                              // ankle to foot: tapers to the model's foot sphere, made in buildBody
const padSkin = new THREE.MeshStandardMaterial({ color: BLACK, roughness: 0.95, metalness: 0.0 });   // the foot's tip: black and matt
const padShape = new THREE.SphereGeometry(1, 12, 10);   // scaled to the model's foot sphere
const legs = [];                                     // built from /api/describe: the world says how many legs
function buildBody(shape, count) {
  prosoma.scale.set(...shape.body);
  seatEyes(shape.body);
  chelicerae.position.set(shape.head_x, 0, 0);
  abdomen.scale.set(...shape.abdomen);               // the model's abdomen and waist: they lie on the floor, never in it
  abdomen.position.set(...shape.abdomen_pos);
  placeCylinder(waist, new THREE.Vector3(...shape.waist[0]), new THREE.Vector3(...shape.waist[1]), shape.waist_radius);
  for (const leg of legs) { spider.remove(leg.coxa, leg.hipJoint, leg.femur, leg.tibia, leg.knee, leg.tarsus, leg.ankle, leg.pad, leg.claw); scene.remove(leg.foot); }
  legs.length = 0;
  tarsusShape = paint(new THREE.CylinderGeometry(shape.foot_radius / TARSUS_R, 1, 1, 12, 8), tarsusColour);   // its tip is the foot sphere
  for (let i = 0; i < count; i++) {
    const femur = new THREE.Mesh(femurShape, bandSkin), tibia = new THREE.Mesh(tibiaShape, bandSkin);
    femur.castShadow = tibia.castShadow = true;
    femur.add(hairs(160, false, 0.3, 0.7, femurColour));
    tibia.add(hairs(200, false, 0.3, 0.8, tibiaColour));
    const tarsus = new THREE.Mesh(tarsusShape, bandSkin);   // the model's tarsus: the foot, a fixed bend at the ankle
    tarsus.castShadow = true;
    tarsus.add(hairs(60, false, 0.3, 0.7, tarsusColour));
    const ankle = new THREE.Mesh(new THREE.SphereGeometry(TIBIA_R * 0.8, 10, 8), ankleSkin);   // rounds the bend
    const knee = new THREE.Mesh(new THREE.SphereGeometry(FEMUR_R * 1.15, 12, 10), kneeSkin);   // the patella
    const coxa = new THREE.Mesh(unit, legSkin);       // body to hip: the model's hip sits at the ellipsoid's rim
    coxa.castShadow = true;
    coxa.add(hairs(60, false, 0.2, 0.5));
    const hipJoint = new THREE.Mesh(new THREE.SphereGeometry(COXA_R, 12, 10), legSkin);
    hipJoint.castShadow = true;
    const pad = new THREE.Mesh(padShape, padSkin);   // the foot's rounded tip: the model's foot sphere, no ball of its own
    pad.scale.setScalar(shape.foot_radius);
    pad.castShadow = true;
    pad.add(hairs(25, true, 0.2, 0.5));
    const claw = new THREE.Mesh(unit, padSkin);     // the model's claw, when the body has one: drawn where the pose puts its tip
    claw.castShadow = true;
    claw.visible = false;
    spider.add(claw);
    const foot = new THREE.Mesh(new THREE.CircleGeometry(0.02, 16), new THREE.MeshBasicMaterial({ color: 0x5ad1ff, transparent: true, opacity: 0.8 }));
    for (const m of [coxa, hipJoint, femur, tibia, knee, tarsus, ankle, pad]) { m.visible = false; spider.add(m); }   // until the first pose places them
    foot.visible = false;
    scene.add(foot);                                 // planted discs lie in the world, on the ground
    legs.push({ coxa, hipJoint, femur, tibia, knee, tarsus, ankle, pad, claw, foot, clawRadius: shape.claw_radius || 0.002 });
  }
}

const target = new THREE.Vector3();
let firstFrame = true;
let lastPose = null;
function draw(pose) {
  lastPose = pose;
  drawArena(pose.arena || 'flat', described && described.terrains, pose.offset);
  spider.position.set(pose.x, pose.y, pose.z);
  const [w, x, y, z] = pose.quat;                    // MuJoCo's order; three.js takes x, y, z, w
  spider.quaternion.set(x, y, z, w);
  spider.updateMatrixWorld();                        // the legs' world points come into the torso's frame
  for (let i = 0; i < legs.length && i < pose.legs.length; i++) {
    const [hx, hy, hz, kx, ky, kz, ax, ay, az, fx, fy, fz, cx, cy, cz] = pose.legs[i];
    const leg = legs[i];
    spider.worldToLocal(_h.set(hx, hy, hz));
    spider.worldToLocal(_a.set(kx, ky, kz));
    spider.worldToLocal(_b.set(fx, fy, fz));
    spider.worldToLocal(_k.set(ax, ay, az));
    // Where the coxa leaves the body. Its own vector: placeCylinder works in _d, so passing _d as
    // the start point would place every coxa by its direction instead (stubs mid-body).
    _r.set(_h.x * 0.85, Math.sign(_h.y) * ROOT_Y, _h.z - 0.004);
    placeCylinder(leg.coxa, _r, _h, COXA_R);
    leg.hipJoint.position.copy(_h);
    placeCylinder(leg.femur, _h, _a, FEMUR_R);
    placeCylinder(leg.tibia, _a, _k, TIBIA_R);
    placeCylinder(leg.tarsus, _k, _b, TARSUS_R);
    leg.ankle.position.copy(_k);
    leg.knee.position.copy(_a);
    leg.pad.position.copy(_b);
    leg.coxa.visible = leg.hipJoint.visible = leg.femur.visible = leg.tibia.visible = leg.knee.visible = leg.tarsus.visible = leg.ankle.visible = leg.pad.visible = true;
    leg.claw.visible = cx !== undefined;            // the pawl's claw, from the foot's centre to its tip
    if (leg.claw.visible) placeCylinder(leg.claw, _b, spider.worldToLocal(_r.set(cx, cy, cz)), leg.clawRadius);
    leg.foot.visible = pose.planted[i];              // a foot on the floor: a disc on the floor under it
    leg.foot.position.set(fx, fy, groundHeight(fx, fy) + 0.003);
  }
  const next = new THREE.Vector3(pose.x, pose.y, groundHeight(pose.x, pose.y) + 0.08);
  if (!viewFollows && !firstFrame) next.copy(target);   // a fixed view: the camera stays where it is, the spider may walk out of it
  if (firstFrame || next.distanceTo(target) > 1.0) {   // a recentre: jump with the spider
    camera.position.sub(target).add(next);
    target.copy(next);
    firstFrame = false;
  } else {
    const before = target.clone();
    target.lerp(next, 0.15);
    camera.position.add(target).sub(before);
  }
  controls.target.copy(target);
  sun.position.set(target.x + 2, target.y - 3, target.z + 5);   // the shadow's box goes with the spider
  sun.target.position.copy(target);
}


function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();
(function frame() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
})();

// --- HUD -----------------------------------------------------------------------------------
const $ = id => document.getElementById(id);
const fmt = (v, d = 3) => (v === null || v === undefined) ? '—' : Number(v).toFixed(d);
// The stance hint in one line: what each count of feet down costs at the values in the fields,
// so the pay and the floor read as the one charge they make (and the stand pay as its other side).
function stanceSum() {
  const pay = Number($('law-stance_pay').value) || 0, floor = Math.round(Number($('law-stance_floor').value) || 4);
  const stand = Number($('law-stand_pay').value) || 0;
  if (!pay) { $('stance-sum').textContent = `stance off · floor ${floor}` + (stand ? `: stand pay +${stand} from ${floor} feet` : ' acts on nothing'); return; }
  const steps = [];
  for (let n = 0; n <= floor; n++) steps.push(`${n}${n === floor && floor < 8 ? '+' : ''}: ${n === floor ? '0' : '−' + fmt(pay * (floor - n), 3)}`);
  $('stance-sum').textContent = `feet → charge  ${steps.join(' · ')}` + (stand ? ` · stand pay +${stand} from ${floor}` : '');
}
for (const id of ['law-stance_pay', 'law-stance_floor', 'law-stand_pay']) $(id).addEventListener('input', stanceSum);
// A number field keeps its own range: a value past min or max is set back to that end when the
// field is left, and when it is read (40 typed into backward pay, a share 0 to 1, becomes 1).
function clamp(field) {
  if (field.type !== 'number' || field.value === '') return;
  let v = Number(field.value);
  if (!Number.isFinite(v)) { field.value = ''; return; }
  if (field.min !== '' && v < Number(field.min)) v = Number(field.min);
  if (field.max !== '' && v > Number(field.max)) v = Number(field.max);
  if (field.step === '1') v = Math.round(v);
  if (String(v) !== field.value) field.value = v;
}
const num = field => { clamp(field); return Number(field.value); };
for (const f of document.querySelectorAll('input[type=number]')) f.addEventListener('change', () => clamp(f));
let described = null;
fetch('/api/describe').then(r => r.json()).then(d => {
  describedBy(d);
  buildBody(d.shape, d.legs);
  if (lastPose) draw(lastPose);                      // a frame may have arrived first (a paused server sends one)
  const b = d.brain;
  $('brainline').textContent = b
    ? `${b.neurons} neurons · ${b.parameters} parameters · ${b.placement.backend}${b.placement.device ? ':' + b.placement.device : ''} ${b.placement.precision} · seed ${d.seed}`
    : `control: ${d.control && d.control.policy}`;
  if (b) openBrainView();
  else $('brainpanel-note').textContent = '· no brain in this life';
}).catch(() => {});
// What the description settles on the page: the law line, the body line and the settings' fields.
// Called again after a control that changes the description (a law change, a load).
let lawFilled = null;                                  // the law the fields were last filled from
let entropyFilled;                                      // the living brain's entropy the field was last filled from
let needFilled;                                         // the living brain's arousal need the field was last filled from
function describedBy(d) {
  described = d;
  if (lastState) $('law').textContent = law(lastState);
  $('body').textContent = `MuJoCo ${d.mujoco} · model ${d.model}` + ({ leg: ', a slot per leg', pair: ', a slot per pair' }[d.layout] || '') + (d.compass ? ', a compass' : '') + (d.ground ? ', the ground sense' : '') + (d.up ? ', the up sense' : '') + (d.load ? ', the load sense' : '');
  const r = d.reward || {};
  // The fields are filled from the server's law only when that law is new to the page: a
  // description re-read for another reason (the socket reconnecting) leaves a half-made edit alone.
  const lawNow = JSON.stringify(LAW_KEYS.map(k => r[k]));
  if (lawNow !== lawFilled) {
    for (const k of LAW_KEYS) {
      const field = $('law-' + k);
      if (field.type === 'checkbox') field.checked = r[k] !== null && r[k] !== undefined && r[k] !== false; else field.value = r[k];
    }
    lawFilled = lawNow;
  }
  stanceSum();
  $('arena').value = r.arena || 'flat';
  if (lastPose) drawArena(r.arena || 'flat', d.terrains, lastPose.offset);
  $('birth-seed').value = d.seed;
  $('birth-layout').value = d.layout || 'pair';
  $('birth-compass').checked = !!d.compass;
  $('birth-ground').checked = !!d.ground;
  $('birth-up').checked = !!d.up;
  $('birth-load').checked = !!d.load;
  const genes = { ...(d.genes || {}) };
  $('birth-hold').value = genes.hold == null ? '' : genes.hold;
  $('birth-entropy').value = genes.entropy == null ? FOUNDER_ENTROPY : genes.entropy;
  // hold=none (not the founder's 0.9) has no number for the field, so it stays in the text
  for (const k of ['hold', 'entropy']) if (!(k === 'hold' && k in genes && genes.hold == null)) delete genes[k];   // maturity and the codes stay in the text: off the page
  $('birth-genes').value = geneWords(genes);
  const retunes = d.gene_changes ? d.gene_changes.length : 0;
  $('genesline').textContent = 'live brain: ' + (geneWords(d.genes) || 'founder genes') + ' · seed ' + d.seed
    + (retunes ? ` · entropy now ${d.entropy}, need now ${d.need} (${retunes} live ${retunes === 1 ? 'change' : 'changes'}: a schedule of genes)` : '');
  // the live entropy field is filled only when the brain's value is new to the page, as the law's
  if (d.entropy !== entropyFilled) {
    $('live-entropy').value = d.entropy === null || d.entropy === undefined ? '' : d.entropy;
    $('live-entropy').disabled = $('apply-entropy').disabled = d.entropy === null || d.entropy === undefined;
    entropyFilled = d.entropy;
  }
  if (d.need !== needFilled) {
    $('live-need').value = d.need === null || d.need === undefined ? '' : d.need;
    $('live-need').disabled = $('apply-need').disabled = d.need === null || d.need === undefined;
    needFilled = d.need;
  }
}
// high_water and old_ground stay law settings (the nursery's --high-water and --old-ground) but are
// not on the page: unmeasured on this body and off; a life born with them still shows its mark.
const FOUNDER_ENTROPY = 0.1;   // spider/brain.py's founder gene (tests/test_server.py holds them equal)
const LAW_KEYS = ['stance_pay', 'yaw_pay', 'window', 'pay_per_metre', 'backward_pay', 'course', 'strict', 'upright_pay', 'stance_upright', 'torque_pay', 'stance_floor', 'stand_pay', 'carry', 'carry_back', 'idle_leg_pay', 'idle_moments', 'idle_height', 'low_pay', 'rear_pay', 'head_pay', 'touchdown_pay'];
// The nursery's words for a brain's gene overrides: {gamma: 0.99, arousal: {need: 0}} → "gamma=0.99 arousal.need=0".
function geneWords(genes, prefix = '') {
  return Object.entries(genes || {}).map(([k, v]) =>
    v && typeof v === 'object' && !Array.isArray(v) ? geneWords(v, prefix + k + '.') : `${prefix}${k}=${Array.isArray(v) ? v.join(',') : v}`
  ).join(' ');
}
async function refreshDescription() {
  try { describedBy(await (await fetch('/api/describe')).json()); } catch (e) { /* the socket shows the state anyway */ }
}

// The reward law, from the settings the server described and the mark of the moment: "progress"
// is the plain law (each moment's progress); a window pays the mean of the last N moments; the
// high-water mark pays only ground beyond the farthest point reached, so the mark is shown with it.
function law(m) {
  const r = described && described.reward;
  if (!r) return '—';
  const parts = [r.window > 1 ? `window ${r.window}` : 'progress'];
  if (r.course !== null && r.course !== undefined) parts.push(`course ${fmt(r.course, 2)} rad${r.strict ? ', strict' : ''} · along it ${fmt(m.distance_course)} m`);
  if (r.pay_per_metre !== 40) parts.push(`${r.pay_per_metre}/m`);
  if (r.backward_pay) parts.push(`backward ${r.backward_pay}`);
  if (r.high_water) parts.push(`high-water${r.old_ground ? ' leaking ' + r.old_ground : ''} · mark ${fmt(m.mark)} m`);
  // the charge a foot and the floor it is short of, both named: "stance 0.1 of 1" read as a share
  if (r.stance_pay) { const floor = r.stance_floor || 4;
    parts.push(`stance ${r.stance_pay}/foot under ${floor} foot${floor === 1 ? '' : 's'}${r.stance_upright === false ? ' (any foot)' : ''}`); }
  if (r.stand_pay) parts.push(`stand ${r.stand_pay}`);
  if (r.carry) parts.push('carried' + (r.carry_back === false ? ', backward charged' : ''));
  if (r.low_pay) parts.push(`low ${r.low_pay}`);
  if (r.rear_pay) parts.push(`rear ${r.rear_pay}`);
  if (r.head_pay) parts.push(`head ${r.head_pay}`);
  if (r.touchdown_pay) parts.push(`touchdown ${r.touchdown_pay}`);
  if (r.idle_leg_pay) parts.push(`idle leg ${r.idle_leg_pay}${r.idle_moments && r.idle_moments !== 20 ? ' after ' + r.idle_moments : ''}`);
  if (r.upright_pay) parts.push(`upright ${r.upright_pay}`);
  if (r.torque_pay) parts.push(`torque ${r.torque_pay}`);
  if (r.yaw_pay) parts.push(`yaw ${r.yaw_pay}`);
  if (r.arena && r.arena !== 'flat') parts.push(r.arena);
  const changes = described.law_changes ? described.law_changes.length : 0;
  if (changes) parts.push(`a schedule: changed ${changes}×`);
  return parts.join(' · ');
}
let lastState = null;

// The speed as a bar under the reading. The moment's speed is a sample of the stride, swinging
// each way within one step, so the bar shows means: the fill is the mean forward speed of the
// last 1000 moments (the learning bar's horizon, 50 s of the body's clock), the tick the last 100.
// The scale is 0.05 m/s each way: the fastest gait measured (1.55 m per 1000 moments) fills 60%.
const SPEED_SCALE = 0.05;
const SPEED_BLOCKS = 10;
function speedbar(m) {
  const dt = (described && described.dt) || 0.05;
  const mean = blocks => blocks.length ? blocks.reduce((a, b) => a + b.forward_m, 0) / (blocks.length * 100 * dt) : 0;
  const share = Math.max(-1, Math.min(1, mean(m.blocks.slice(-SPEED_BLOCKS)) / SPEED_SCALE)) * 50;
  const fill = $('speed-fill');
  fill.classList.toggle('back', share < 0);
  fill.style.left = (share < 0 ? 50 + share : 50) + '%';
  fill.style.width = Math.abs(share) + '%';
  $('speed-mean').style.left = (50 + Math.max(-1, Math.min(1, mean(m.blocks.slice(-1)) / SPEED_SCALE)) * 50) + '%';
  $('speed-label').textContent = fmt(mean(m.blocks.slice(-SPEED_BLOCKS))) + ' m/s';
}

// --- the brain view: the whole connectome in three dimensions, on the right -----------------
// The view is the connectome demo's (static/vendor/brain_view.js); the payload is this brain's
// connectome laid out as an ellipsoid, senses at the back, motor cortex at the front, each
// neuron's lines its incoming synapses. Every frame paints the settled state blue and, on a learning
// moment, flashes gold where incoming synapses changed; the flash fades over the next frames.
let brainView = null, flash = null;
async function openBrainView() {
  try {
    const payload = await (await fetch('/api/brainview')).json();
    if (!payload || brainView) return;
    brainView = new BrainView($('view-host'), payload, {
      // some forty fibres cross every pixel of the ball, so each one is faint and the tone
      // map's exposure low; the demo's colours would burn the ball white
      spin: true, controls: 'legend', scale: 'auto', gamma: 0.8, floor: 0.05, exposure: 0.09,
      rest: [0.002, 0.005, 0.012], lit: [0.05, 0.13, 0.32], flash: [0.6, 0.22, 0.01],   // blue for the state, gold for learning
      somaRadius: 0.012, pointRadius: 0.8, pointDim: 0.35, fitScale: 0.95,
    });
    flash = new Float32Array(payload.order.length);
    brainView.start();
    $('brainpanel-note').textContent = `· ${payload.counts.neurons} neurons, ${payload.counts.synapses} synapses`;
    window.addEventListener('resize', () => brainView.resize());
    $('view-fit').addEventListener('click', () => brainView.fit());
    $('view-spin').addEventListener('click', () => {
      brainView.options.spin = !brainView.options.spin;
      $('view-spin').classList.toggle('on', brainView.options.spin);
    });
  } catch (e) {
    $('brainpanel-note').textContent = '· view unavailable';
  }
}
const FLASH_FADE = 0.96;                              // per frame received, as the demo's page
function paintBrain(m) {
  if (!brainView) return;
  const a = m.activity;
  if (!a || !a.all) return;                           // the random control: the last brain state stays
  const repair = a.repair;
  for (let i = 0; i < flash.length; i++) {
    const f = flash[i] * FLASH_FADE;
    flash[i] = repair && repair[i] > f ? repair[i] : f < 0.01 ? 0 : f;
  }
  brainView.setActivity(a.all, flash);
}

// The wait until the spider moves: the page opens with it shown (the server may still be
// composing or loading the brain), a Reset and a lost connection show it again, and it goes
// once the moments count on, or at once on a paused life, which will not move until resumed.
let initFrom = null;
function initializing(on) {
  $('init').classList.toggle('show', on);
  if (on) initFrom = null;
}
function settled(m) {
  if (!$('init').classList.contains('show')) return;
  if (m.paused) { initializing(false); return; }
  if (initFrom === null) initFrom = m.moments;
  else if (m.moments > initFrom) initializing(false);
}

function update(m) {
  settled(m);
  draw(m.pose);
  $('policy').textContent = m.policy;
  $('moments').textContent = m.moments;
  const mode = $('mode');
  mode.textContent = m.refused ? 'refused' : m.mode;
  mode.className = m.refused ? 'refused' : m.mode;
  $('forward').textContent = fmt(m.distance_forward) + ' m';
  $('net').textContent = fmt(m.displacement_m) + ' m' + (m.recentres ? ` · recentred ${m.recentres}×` : '');
  $('speed').textContent = fmt(m.speed) + ' m/s';
  speedbar(m);
  $('reward').textContent = fmt(m.reward, 3) + (m.owed !== null && m.owed !== undefined ? ` (owed ${fmt(m.owed, 3)})` : '');
  lastState = m;
  $('law').textContent = law(m);
  // a want of 1 is the whole of it: the recent reward is at or below nothing the brain could need,
  // so the arousal law cannot calm whatever the need is set to (cadence arousal.py's want)
  const wantFull = m.want !== null && m.want !== undefined && m.want >= 0.99;
  $('arousal').textContent = `${fmt(m.level, 2)} / ${fmt(m.want, 2)}` + (wantFull ? ' · full' : '');
  $('arousal').className = 'v' + (wantFull ? ' warn' : '');
  $('sweeps').textContent = `${m.sweeps} (${m.learning_sweeps} learning)`;
  $('ms').textContent = fmt(m.ms, 1);
  $('pace').textContent = (m.pace > 0 ? `${m.pace}/s` : 'max') + (m.rate > 0 ? ` · ${fmt(m.rate, 0)}/s made` : '');
  const up = $('upright');
  up.textContent = fmt(m.upright, 2) + (m.upright > ((described && described.upright) || 0.5) ? '' : ' · fallen');
  up.className = 'v' + (m.upright > ((described && described.upright) || 0.5) ? '' : ' fallen');
  $('tilt').textContent = m.tilt ? `${fmt(m.tilt[0] * 180 / Math.PI, 0)}° / ${fmt(m.tilt[1] * 180 / Math.PI, 0)}°` : '—';
  $('falls').textContent = (m.upright_share === null || m.upright_share === undefined ? '—' : Math.round(100 * m.upright_share) + '%') + ` · ${m.falls} fall${m.falls === 1 ? '' : 's'}`;
  $('stance').textContent = m.pose.planted.filter(Boolean).length + ' / ' + m.pose.planted.length;
  const paid = m.blocks.filter(b => b.reward_per_moment !== undefined && b.reward_per_moment !== null);
  const lastPaid = paid.length ? paid[paid.length - 1].reward_per_moment : null;
  const meanPaid = paid.length ? paid.reduce((a, b) => a + b.reward_per_moment, 0) / paid.length : null;
  $('reward-mean').textContent = paid.length
    ? `${fmt(meanPaid, 3)} over ${paid.length * 100} · last ${fmt(lastPaid, 3)}`
      + (lastPaid <= -0.499 ? ' · cap' : meanPaid < 0 ? ' · below 0' : '')
    : '—';
  $('reward-mean').className = 'v' + (paid.length && (lastPaid <= -0.499 || meanPaid < 0) ? ' warn' : '');
  const termed = m.blocks.filter(b => b.terms);
  const TERM_NAMES = ['progress', 'stand', 'touchdown', 'idle', 'low', 'rear', 'head', 'stance', 'yaw', 'upright', 'torque'];
  const termMean = k => termed.reduce((a, b) => a + (b.terms[k] || 0), 0) / termed.length;
  const shown = termed.length ? TERM_NAMES.filter(k => k === 'progress' || termed.some(b => b.terms[k])) : [];
  // what the law gives against what it takes, before the cap: a charge larger than every gift
  // leaves no act that pays, only acts that cost less
  const given = termed.length ? TERM_NAMES.reduce((a, k) => a + Math.max(0, termMean(k)), 0) : 0;
  const taken = termed.length ? TERM_NAMES.reduce((a, k) => a + Math.min(0, termMean(k)), 0) : 0;
  $('reward-balance').textContent = termed.length
    ? `given +${fmt(given, 3)} · taken ${fmt(taken, 3)}` + (given + taken < 0 ? ' · the charges outweigh every gift' : '')
    : '—';
  $('reward-balance').className = 'v' + (termed.length && given + taken < 0 ? ' warn' : '');
  $('reward-terms').textContent = termed.length
    ? shown.map(k => `${k} ${fmt(termMean(k), 3)}`).join(' · ') + ` (over ${termed.length * 100})`
      + (termed.some(b => b.touchdowns !== undefined) ? ` · ${termed.reduce((a, b) => a + (b.touchdowns || 0), 0)} touchdowns` : '')
    : '—';
  const r = described && described.reward;
  const worst = r && r.worst_charge !== undefined ? r.worst_charge : null;
  const cap = (r && r.cap) || 0.5;
  const shared = m.blocks.filter(b => b.capped_share !== undefined);
  const cappedShare = shared.length ? shared.reduce((a, b) => a + b.capped_share, 0) / shared.length : null;
  $('worst-charge').textContent = (worst === null ? '—' : `${fmt(worst, 3)} of ${fmt(cap, 2)}${worst >= cap ? ' · can reach the cap' : ''}`)
    + (cappedShare === null ? '' : ` · capped ${Math.round(100 * cappedShare)}% of ${shared.length * 100}`);
  $('worst-charge').className = 'v' + ((worst !== null && worst >= cap) || cappedShare > 0.05 ? ' warn' : '');
  const stanced = m.blocks.filter(b => b.mean_stance !== undefined && b.mean_stance !== null);
  $('mean-stance').textContent = stanced.length
    ? `${fmt(stanced.reduce((a, b) => a + b.mean_stance, 0) / stanced.length, 2)} over ${stanced.length * 100} · last 100: ${fmt(stanced[stanced.length - 1].mean_stance, 2)}`
    : '—';
  $('turned').textContent = `${fmt(m.turned, 1)} rad (${fmt(m.omega, 2)} rad/s)`;
  $('refusals').textContent = m.refusals;
  $('pause').textContent = m.paused ? 'Resume' : 'Pause';
  $('pause').dataset.op = m.paused ? 'resume' : 'pause';
  for (const b of document.querySelectorAll('button[data-op=policy]')) b.classList.toggle('on', b.dataset.value === m.policy);
  for (const b of document.querySelectorAll('button[data-op=pace]')) b.classList.toggle('on', Number(b.dataset.value) === m.pace);
  const blocks = m.blocks.slice(-4);
  $('blocks').textContent = blocks.length
    ? 'last blocks (100 moments): ' + blocks.map(b => `${b.forward_m >= 0 ? '+' : ''}${b.forward_m.toFixed(3)} m, ${b.reward_per_moment.toFixed(3)} r`).join(' · ')
    : '';
  const notes = $('notes');
  const pinned = notes.scrollHeight - notes.scrollTop - notes.clientHeight < 4;   // at the end, or not yet scrolling
  notes.textContent = m.notes.join('\n');
  if (pinned) notes.scrollTop = notes.scrollHeight;
  learning(m);
  strip(m);
  paintBrain(m);
}

// --- the learning bar: 0 "Brain Stall" to 100 "Learning" ------------------------------------
// The brain learns only while aroused (routine moments are greedy and change nothing), so the
// bar is the aroused share of the brain's last LEARNING_BLOCKS blocks, 0-100. Before the first
// block closes it follows the mode of the moment. Random and frozen never learn: the bar stays
// empty and the note says why. It is a reading of the arousal law, not a measure of a gait.
const LEARNING_BLOCKS = 10;
function learning(m) {
  const pct = $('learning-pct'), cover = $('learning-cover'), note = $('learning-note');
  let score = null, text = '';
  if (m.policy !== 'brain') {
    score = 0;
    text = m.policy === 'random' ? 'the random control never learns' : 'the frozen copy never learns';
  } else {
    const blocks = m.blocks.filter(b => b.policy === 'brain').slice(-LEARNING_BLOCKS);
    if (blocks.length) {
      const share = blocks.reduce((s, b) => s + b.aroused_share, 0) / blocks.length;
      const sweeps = blocks.reduce((s, b) => s + b.learning_sweeps, 0);
      score = Math.round(100 * share);
      text = `aroused in ${score}% of the last ${blocks.length * 100} moments · ${sweeps} learning sweeps`;
    } else {
      score = m.mode === 'aroused' ? 100 : 0;
      text = 'no block closed yet · following the mode of the moment';
    }
  }
  pct.textContent = score;
  cover.style.width = (100 - score) + '%';
  note.textContent = text;
}

// --- the brain strip: sensory, association and motor activity of the moment ----------------
const brainCanvas = $('brain'), ctx = brainCanvas.getContext('2d');
function strip(m) {
  const W = brainCanvas.width, H = brainCanvas.height;
  ctx.clearRect(0, 0, W, H);
  const a = m.activity;
  if (!a) {
    ctx.fillStyle = '#8b94a7';
    ctx.font = '11px system-ui';
    ctx.fillText(m.policy === 'random' ? 'random policy: no brain' : 'no settled state', 8, 20);
    return;
  }
  row(a.sensory, 7, 26, W - 8, 1);                  // each label sits above its row with a clear gap
  row(a.association, 43, 20, W - 8, 1);
  // One column per joint, as many as the body has (twenty-four), with its three answers. Under a
  // slot per joint the motor units are the joint's own; under a slot per leg or pair they are the
  // slot's joint answers (27 each), not the joint's, so their row is drawn above the joints' commands.
  const motor = a.motor, joints = m.commands ? m.commands.length : motor.length / 3;
  const perJoint = motor.length === 3 * joints, cell = (W - 8) / joints;
  const top = perJoint ? 76 : 85, h = perJoint ? 24 : 15;
  if (!perJoint) row(motor, 68, 15, W - 8, 0);
  for (let j = 0; j < joints; j++) {
    for (let k = 0; k < 3; k++) {
      ctx.fillStyle = perJoint ? heat(motor[3 * j + k]) : '#1b2230';
      ctx.fillRect(4 + j * cell, top + k * (h / 3), cell - 1, h / 3 - 1);
    }
    if (m.commands) {
      ctx.strokeStyle = '#5ad1ff';
      ctx.strokeRect(4 + j * cell + 0.5, top + m.commands[j] * (h / 3) + 0.5, cell - 2, h / 3 - 2);
    }
  }
  ctx.fillStyle = '#8b94a7';
  ctx.font = '9px system-ui';
  ctx.fillText('sensory', 6, 10); ctx.fillText('association', 6, 46);
  ctx.fillText(perJoint ? `motor: to low end · hold setpoint · to high end, per joint (${joints})`
    : `motor (${motor.length}) · each joint's command (${joints}): low end · hold · high end`, 6, 72);
}
function row(values, y, h, w, pad) {
  const cell = w / values.length;
  for (let i = 0; i < values.length; i++) {
    ctx.fillStyle = heat(values[i]);
    ctx.fillRect(4 + i * cell, y + 8, Math.max(1, cell - pad), h - 8);
  }
}
function heat(v) {
  const t = Math.max(0, Math.min(1, v));
  return `rgb(${Math.round(20 + 235 * t)}, ${Math.round(40 + 160 * t)}, ${Math.round(70 + 40 * (1 - t))})`;
}

// --- controls and the socket ---------------------------------------------------------------
// Reset gives birth to what the settings name; genes are composed once, so this is the one
// place they are set. The page reloads on the new life: a reset is a new life, not a new frame.
function birth() {
  // hold and entropy have fields of their own; the rest (maturity and the codes among it) is the text, in the nursery's words
  const words = [$('birth-genes').value.trim()];
  if ($('birth-hold').value !== '') words.push('hold=' + num($('birth-hold')));
  // entropy is sent whenever it is not the founder's, 0 included: an empty word would be the founder's 0
  if ($('birth-entropy').value !== '' && Number($('birth-entropy').value) !== FOUNDER_ENTROPY) words.push('entropy=' + Number($('birth-entropy').value));
  return { seed: num($('birth-seed')), layout: $('birth-layout').value,
           compass: $('birth-compass').checked, ground: $('birth-ground').checked,
           // up only when it matters (ticked, or a life that has it): a server older than the up
           // sense refuses a birth that names it, and a birth that leaves it out keeps the life's
           ...($('birth-up').checked || (described && described.up) ? { up: $('birth-up').checked } : {}),
           ...($('birth-load').checked || (described && described.load) ? { load: $('birth-load').checked } : {}),
           genes: words.filter(Boolean).join(' ') };
}
function resetQuestion(b) {
  return 'Reset everything?\n\nA newborn brain replaces the live one: seed ' + b.seed + ({ leg: ', a slot per leg', pair: ', a slot per pair' }[b.layout] || '')
    + (b.compass ? ', a compass' : '') + (b.ground ? ', the ground sense' : '') + (b.up ? ', the up sense' : '') + (b.load ? ', the load sense' : '') + ', ' + (b.genes ? 'genes ' + b.genes : 'founder genes') + '.\n'
    + 'The world returns to the origin, every counter goes to zero and the saved life in runs/ is deleted.\n\n'
    + 'The brain that has lived until now is gone for good.';
}
// A word that hovers over a button for a moment and fades: the page's answer to a click
// whose result shows nowhere else (a save changes nothing on the panel).
function toast(anchor, text) {
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = text;
  document.body.appendChild(t);
  const r = anchor.getBoundingClientRect();
  t.style.left = (r.left + r.width / 2) + 'px';
  t.style.top = r.top + 'px';
  requestAnimationFrame(() => t.classList.add('show'));
  setTimeout(() => { t.classList.remove('show'); setTimeout(() => t.remove(), 300); }, 1100);
}
let lastRefusal = '';
async function control(body) {
  const r = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) { lastRefusal = await r.text(); $('notes').textContent = lastRefusal; return null; }
  return r.json();
}
document.querySelectorAll('button[data-op]').forEach(b => b.addEventListener('click', async () => {
  const body = { op: b.dataset.op };
  if (b.dataset.op === 'reset') {
    body.value = birth();
    if (!window.confirm(resetQuestion(body.value))) return;
    initializing(true);
  } else if (b.dataset.value !== undefined) body.value = isNaN(b.dataset.value) ? b.dataset.value : Number(b.dataset.value);
  const result = await control(body);
  if (!result && b.dataset.op === 'reset') {                       // refused: the old life goes on, and says why
    initializing(false);
    window.alert('No birth: ' + lastRefusal + '\nThe old life goes on.');
  }
  if (result && b.dataset.op === 'reset') location.reload();
  else if (result && b.dataset.op === 'load') refreshDescription();
  else if (result && b.dataset.op === 'save') toast(b, result.saved ? 'saved' : (result.why || 'not saved'));
}));
async function applyLaw() {
  const value = {};
  for (const k of LAW_KEYS) {
    const field = $('law-' + k);
    value[k] = field.type === 'checkbox' ? field.checked : num(field);
  }
  // the course box: ticked on a life with a course keeps that heading; ticked on one without, the
  // server takes the heading the body holds now (true); unticked is off (null)
  const had = described && described.reward ? described.reward.course : null;
  value.course = value.course ? (had === null || had === undefined ? true : had) : null;
  if (await control({ op: 'law', value })) refreshDescription();
}
// the law's fields stand in two columns, each with its own button: both apply the whole law
$('apply-law').addEventListener('click', applyLaw);
$('apply-law-progress').addEventListener('click', applyLaw);
// The arena select: the ground is the world's law, unlike the colours below it, so a change is a
// law change on the living brain and makes the life a schedule.
$('arena').addEventListener('change', async (e) => {
  const was = (described && described.reward && described.reward.arena) || 'flat';
  if (await control({ op: 'law', value: { arena: e.target.value } })) refreshDescription();
  else e.target.value = was;
});
$('apply-entropy').addEventListener('click', async () => {
  if (await control({ op: 'entropy', value: num($('live-entropy')) })) refreshDescription();
});
$('apply-need').addEventListener('click', async () => {
  if (await control({ op: 'need', value: num($('live-need')) })) refreshDescription();
});

function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => { $('link').textContent = 'live'; if (described) refreshDescription(); };   // a restarted server may hold another life
  ws.onmessage = e => update(JSON.parse(e.data));
  ws.onclose = () => { $('link').textContent = 'reconnecting…'; initializing(true); setTimeout(connect, 1000); };
  ws.onerror = () => ws.close();
}
connect();

// --- the world panel: the surface and the colours of the ground and the grid, how the world is drawn, not its law
// (the brain senses joints, feet and motion, never a colour; the choices live in this browser only)
const VIEW = {
  ground: { own: GROUND_COLOUR, field: 'ground-colour', paint: hex => setGroundColour(hex) },
  surface: { own: 'plain', field: 'surface', paint: name => setSurface(name) },
  follow: { own: 'follows', field: 'follow', paint: value => setView(value) },
  grid: { own: GRID_COLOUR, field: 'grid-colour', paint: hex => makeGrid(hex) },
  light: { own: '1', field: 'light', paint: v => { sky.intensity = LIGHT.sky * Number(v); sun.intensity = LIGHT.sun * Number(v); } },
};
function paintView(name, value) {
  VIEW[name].paint(value);
  $(VIEW[name].field).value = value;
  if (name === 'light') $('light-value').textContent = Number(value).toFixed(1) + '×';
}
function rememberView(name, value) {
  const key = 'world-' + name;
  try { if (String(value) === VIEW[name].own) localStorage.removeItem(key); else localStorage.setItem(key, value); } catch (e) { /* no storage: the choice lasts the page */ }
}
for (const name of Object.keys(VIEW)) {
  paintView(name, VIEW[name].own);
  try { const kept = localStorage.getItem('world-' + name); if (kept) paintView(name, kept); } catch (e) { /* no storage */ }
  $(VIEW[name].field).addEventListener('input', (e) => { paintView(name, e.target.value); rememberView(name, e.target.value); });
}
$('view-default').addEventListener('click', () => {
  for (const name of Object.keys(VIEW)) { paintView(name, VIEW[name].own); rememberView(name, VIEW[name].own); }
});
