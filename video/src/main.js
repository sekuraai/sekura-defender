// One master anime.js timeline. The renderer seeks it frame by frame (window.__seek), so every
// visual must be driven by the timeline -- no CSS transitions, no timers, or renders go non-deterministic.
import { createTimeline, stagger, utils } from './vendor/anime.esm.js';
import { SCENES, DURATION } from './content.js';
import * as s0 from './scenes/s0.js';
import * as s1 from './scenes/s1.js';
import * as s2 from './scenes/s2.js';
import * as s3 from './scenes/s3.js';
import * as s4 from './scenes/s4.js';
import * as s5 from './scenes/s5.js';
import * as s6 from './scenes/s6.js';
import * as s7 from './scenes/s7.js';
import * as s8 from './scenes/s8.js';

const MODULES = { s0, s1, s2, s3, s4, s5, s6, s7, s8 };
const tl = createTimeline({ autoplay: false, defaults: { ease: 'outQuint', duration: 600 } });
const R = Math.round;
const fmtInt = (n) => R(n).toLocaleString('en-US');
const fmt3 = (n) => n.toFixed(3);

function makeCtx(scene) {
  const root = document.getElementById(scene.id);
  const $ = (s) => root.querySelector(s);
  const $$ = (s) => [...root.querySelectorAll(s)];
  const add = (t, p, at) => { tl.add(t, p, R(at)); };
  const set = (t, p, at) => { tl.set(t, p, R(at)); };
  const counter = (el, to, at, dur = 1400, fmt = fmtInt, from = 0, ease = 'outExpo') => {
    const o = { v: from }; el.textContent = fmt(from);
    tl.add(o, { v: to, duration: dur, ease, onUpdate: () => { el.textContent = fmt(o.v); } }, R(at));
  };
  const type = (el, text, at, cps = 34) => {
    const o = { n: 0 }; const dur = (text.length * 1000) / cps; el.textContent = '';
    tl.add(o, { n: text.length, duration: dur, ease: 'linear', onUpdate: () => { el.textContent = text.slice(0, R(o.n)); } }, R(at));
    return at + dur;
  };
  // Fade in, hold, fade out. `t` is relative to the scene start everywhere in a scene module.
  const frame = ({ fadeIn = 420, fadeOut = 420 } = {}) => {
    add(root, { opacity: [0, 1], duration: fadeIn, ease: 'linear' }, scene.start);
    add(root, { opacity: [1, 0], duration: fadeOut, ease: 'linear' }, scene.start + scene.dur - fadeOut);
  };
  const t = (ms) => scene.start + ms;
  return { root, $, $$, add, set, counter, type, frame, t, S: scene, tl, stagger, utils, fmt3, fmtInt };
}

for (const scene of SCENES) {
  const ctx = makeCtx(scene);
  ctx.root.innerHTML = MODULES[scene.id].html();
  MODULES[scene.id].build(ctx);
}
tl.call(() => {}, DURATION);

window.__duration = DURATION;
window.__cues = [];
window.__seek = (ms) => { tl.seek(ms); };
await document.fonts.ready;
// Prime: run to the end and back so every tween has written its "from" state.
tl.seek(DURATION); tl.seek(0);
window.__ready = true;

const params = new URLSearchParams(location.search);
if (params.has('preview')) {
  document.body.classList.add('preview');
  const stage = document.getElementById('stage');
  const fit = () => { const s = Math.min(innerWidth / 1920, (innerHeight - 48) / 1080); stage.style.transform = `scale(${s})`; };
  addEventListener('resize', fit); fit();
  const scrub = document.getElementById('scrub'); const clock = document.getElementById('clock');
  const playBtn = document.getElementById('play'); scrub.max = DURATION;
  let cur = Number(params.get('t') || 0), playing = false, last = 0;
  const show = () => { tl.seek(cur); scrub.value = cur; clock.textContent = `${Math.floor(cur / 60000)}:${((cur % 60000) / 1000).toFixed(1).padStart(4, '0')}`; };
  scrub.oninput = () => { cur = Number(scrub.value); show(); };
  playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? '❚❚' : '▶'; last = performance.now(); };
  document.getElementById('jumps').innerHTML = SCENES.map((s) => `<button data-t="${s.start}">${s.id}</button>`).join('');
  document.getElementById('jumps').onclick = (e) => { if (e.target.dataset.t) { cur = Number(e.target.dataset.t); show(); } };
  addEventListener('keydown', (e) => { if (e.code === 'Space') { e.preventDefault(); playBtn.click(); } });
  const loop = (now) => { if (playing) { cur = Math.min(DURATION, cur + (now - last)); show(); } last = now; requestAnimationFrame(loop); };
  requestAnimationFrame(loop); show();
}
