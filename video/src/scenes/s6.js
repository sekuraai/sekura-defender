import { COPY } from '../content.js';
const C = COPY.s6;
export const html = () => `
  <div class="abs" style="left:140px;top:170px;width:1640px">
    <div id="k" class="kicker" style="opacity:0">${C.kicker}</div>
    <div id="b" class="body2" style="margin-top:30px;max-width:1340px;opacity:0">${C.body}</div>
    <div class="row" style="margin-top:78px;gap:120px;align-items:flex-start">
      <div>
        <div class="lbl" style="font-size:24px">planner score</div>
        <div class="row" style="margin-top:18px;gap:28px;align-items:baseline">
          <span class="mono" style="font:400 44px var(--mono);color:var(--term-gray)">0.600</span>
          <span class="mono" style="font:400 36px var(--mono);color:var(--term-dim)">&rarr;</span>
          <span id="p1" class="mono" style="font:800 92px var(--mono);background:var(--grad-text);-webkit-background-clip:text;background-clip:text;color:transparent;font-variant-numeric:tabular-nums">0.000</span>
        </div>
      </div>
      <div>
        <div class="lbl" style="font-size:24px">zero-coverage rate</div>
        <div class="row" style="margin-top:18px;gap:28px;align-items:baseline">
          <span class="mono" style="font:400 44px var(--mono);color:var(--term-finding)">20%</span>
          <span class="mono" style="font:400 36px var(--mono);color:var(--term-dim)">&rarr;</span>
          <span id="p2" class="mono" style="font:800 92px var(--mono);color:var(--signal);font-variant-numeric:tabular-nums">20%</span>
        </div>
      </div>
    </div>
    <div id="f" class="mono" style="margin-top:84px;max-width:1420px;font:500 29px/1.45 var(--mono);color:var(--term-fg);opacity:0">${C.foot}</div>
  </div>`;
export const build = (c) => {
  c.frame();
  c.add(c.$('#k'), { opacity: [0, 1], x: [-14, 0] }, c.t(150));
  c.add(c.$('#b'), { opacity: [0, 1], duration: 700 }, c.t(420));
  c.counter(c.$('#p1'), 1.0, c.t(2200), 1500, c.fmt3, 0.6);
  const o = { v: 20 };
  c.add(o, { v: 0, duration: 1500, ease: 'outExpo', onUpdate: () => { c.$('#p2').textContent = `${Math.round(o.v)}%`; } }, c.t(2600));
  c.add(c.$('#f'), { opacity: [0, 1], duration: 760 }, c.t(4600));
};
