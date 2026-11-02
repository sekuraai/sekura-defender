import { COPY } from '../content.js';
const C = COPY.s1;
export const html = () => `
  <div class="abs" style="left:140px;top:150px;width:1640px">
    <div id="k" class="kicker" style="opacity:0">${C.kicker}</div>
    <div id="h" class="h2" style="margin-top:28px;max-width:1340px;opacity:0">${C.head}</div>
    <div id="b" class="body2" style="margin-top:34px;max-width:1180px;opacity:0">${C.body}</div>
    <div class="row" style="margin-top:62px;gap:16px;flex-wrap:wrap;max-width:1500px">
      ${C.jobs.map((j) => `<span class="pillbox job" style="opacity:0">${j}</span>`).join('')}
    </div>
    <div id="cost" class="mono" style="margin-top:56px;font:600 30px var(--mono);color:var(--term-finding);opacity:0">
      narrow · schema-constrained · repetitive — and billed at frontier rates
    </div>
  </div>`;
export const build = (c) => {
  c.frame();
  c.add(c.$('#k'), { opacity: [0, 1], x: [-14, 0] }, c.t(150));
  c.add(c.$('#h'), { opacity: [0, 1], y: [18, 0], duration: 800 }, c.t(450));
  c.add(c.$('#b'), { opacity: [0, 1], duration: 700 }, c.t(1100));
  c.add(c.$$('.job'), { opacity: [0, 1], y: [16, 0], duration: 560, delay: c.stagger(170) }, c.t(2300));
  c.add(c.$('#cost'), { opacity: [0, 1], duration: 700 }, c.t(4100));
};
