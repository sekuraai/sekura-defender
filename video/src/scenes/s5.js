import { COPY, TASKS, OVERALL } from '../content.js';
const C = COPY.s5;
const SCALE = 1020;           // px per 1.000
const ROW = 48;
const TOP = 286;
export const html = () => `
  <div class="abs" style="left:140px;top:120px;width:1680px">
    <div id="k" class="kicker" style="opacity:0">${C.kicker}</div>
    <div id="h" class="body2" style="margin-top:20px;font-size:27px;opacity:0">${C.head}</div>
    <div class="row" style="margin-top:28px;gap:34px;opacity:0" id="leg">
      <span class="row" style="gap:12px"><span style="width:30px;height:11px;border-radius:3px;background:#39433f"></span><span class="lbl" style="font-size:23px">base (untuned)</span></span>
      <span class="row" style="gap:12px"><span style="width:30px;height:17px;border-radius:3px;background:linear-gradient(90deg,#1a5fd6,#35c0f0)"></span><span class="lbl" style="font-size:23px">rank-32 LoRA</span></span>
    </div>
  </div>
  <div class="abs" style="left:140px;top:${TOP}px;width:1680px">
    ${TASKS.map(([name, b, t], i) => {
      const reg = t < b;
      return `<div class="trow row" style="height:${ROW}px;gap:20px;opacity:0">
        <span class="mono" style="width:300px;font:500 25px var(--mono);color:${reg ? 'var(--term-exploit)' : 'var(--term-gray)'}">${name}</span>
        <span style="position:relative;width:${SCALE}px;height:36px">
          <span class="bb" style="position:absolute;left:0;top:2px;height:11px;border-radius:3px;background:#39433f;width:0" data-w="${(b * SCALE).toFixed(1)}"></span>
          <span class="bt" style="position:absolute;left:0;top:18px;height:17px;border-radius:3px;width:0;background:${reg ? '#FF7A7A' : 'linear-gradient(90deg,#1a5fd6,#35c0f0)'}" data-w="${(t * SCALE).toFixed(1)}"></span>
        </span>
        <span class="mono tn" style="width:120px;text-align:right;font:700 27px var(--mono);color:#fff;font-variant-numeric:tabular-nums">0.000</span>
        <span class="mono td" style="width:120px;text-align:right;font:600 24px var(--mono);color:${reg ? 'var(--term-exploit)' : 'var(--signal)'};font-variant-numeric:tabular-nums">&nbsp;</span>
      </div>`;
    }).join('')}
  </div>
  <div class="abs" id="ov" style="left:140px;top:${TOP + TASKS.length * ROW + 26}px;width:1680px;opacity:0">
    <div style="height:2px;background:rgba(215,221,217,.14)"></div>
    <div class="row" style="margin-top:22px;gap:20px;align-items:baseline">
      <span class="mono" style="width:300px;font:700 30px var(--mono);color:#fff">overall</span>
      <span class="mono" style="font:400 28px var(--mono);color:var(--term-gray)">0.4690</span>
      <span class="mono" style="font:400 28px var(--mono);color:var(--term-dim)">&rarr;</span>
      <span id="ovt" class="mono" style="font:800 52px var(--mono);background:var(--grad-text);-webkit-background-clip:text;background-clip:text;color:transparent;font-variant-numeric:tabular-nums">0.0000</span>
      <span id="ovd" class="mono" style="font:700 34px var(--mono);color:var(--signal);opacity:0">+0.1963</span>
    </div>
  </div>`;
export const build = (c) => {
  c.frame();
  c.add(c.$('#k'), { opacity: [0, 1], x: [-14, 0] }, c.t(150));
  c.add(c.$('#h'), { opacity: [0, 1], duration: 600 }, c.t(420));
  c.add(c.$('#leg'), { opacity: [0, 1], duration: 500 }, c.t(800));
  c.$$('.trow').forEach((el, i) => {
    const at = c.t(1300 + i * 560);
    const [, b, t] = TASKS[i];
    c.add(el, { opacity: [0, 1], duration: 260 }, at);
    c.add(el.querySelector('.bb'), { width: [0, Number(el.querySelector('.bb').dataset.w)], duration: 520 }, at);
    c.add(el.querySelector('.bt'), { width: [0, Number(el.querySelector('.bt').dataset.w)], duration: 760 }, at + 300);
    c.counter(el.querySelector('.tn'), t, at + 300, 760, c.fmt3);
    const d = t - b;
    c.type(el.querySelector('.td'), `${d >= 0 ? '+' : ''}${d.toFixed(3)}`, at + 900, 26);
  });
  const ovAt = c.t(1300 + TASKS.length * 560 + 400);
  c.add(c.$('#ov'), { opacity: [0, 1], duration: 520 }, ovAt);
  c.counter(c.$('#ovt'), OVERALL.tuned, ovAt + 200, 1300, (n) => n.toFixed(4));
  c.add(c.$('#ovd'), { opacity: [0, 1], scale: [0.9, 1], duration: 520 }, ovAt + 1500);
};
