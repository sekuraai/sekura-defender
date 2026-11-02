import { COPY } from '../content.js';
const C = COPY.s2;
export const html = () => `
  <div class="abs" style="left:140px;top:140px;width:1640px">
    <div id="k" class="kicker" style="opacity:0">${C.kicker}</div>
    <div id="h" class="h2" style="margin-top:28px;max-width:1420px;opacity:0">${C.head}</div>
    <div style="margin-top:66px;display:flex;flex-direction:column;gap:26px">
      ${C.items.map((it, i) => `
        <div class="vrow row" style="gap:28px;opacity:0">
          <div style="width:10px;height:66px;border-radius:3px;background:linear-gradient(180deg,#1a5fd6,#35c0f0)"></div>
          <div>
            <div class="mono" style="font:700 36px var(--mono);color:#fff">${it[0]}</div>
            <div class="lbl" style="margin-top:8px;font-size:25px">${it[1]}</div>
          </div>
          <div class="mono vck" style="margin-left:auto;font:700 40px var(--mono);color:var(--signal);opacity:0">&#10003;</div>
        </div>`).join('')}
    </div>
    <div id="f" class="body2" style="margin-top:62px;max-width:1240px;color:var(--term-fg);opacity:0">${C.foot}</div>
  </div>`;
export const build = (c) => {
  c.frame();
  c.add(c.$('#k'), { opacity: [0, 1], x: [-14, 0] }, c.t(150));
  c.add(c.$('#h'), { opacity: [0, 1], y: [18, 0], duration: 800 }, c.t(420));
  c.$$('.vrow').forEach((el, i) => {
    c.add(el, { opacity: [0, 1], x: [-20, 0], duration: 620 }, c.t(1500 + i * 900));
    c.add(el.querySelector('.vck'), { opacity: [0, 1], scale: [0.6, 1], duration: 420 }, c.t(1500 + i * 900 + 480));
  });
  c.add(c.$('#f'), { opacity: [0, 1], duration: 760 }, c.t(4700));
};
