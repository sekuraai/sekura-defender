import { COPY } from '../content.js';
const C = COPY.s3;
const W = [1180, 760, 420, 330];
export const html = () => `
  <div class="abs" style="left:140px;top:140px;width:1640px">
    <div id="k" class="kicker" style="opacity:0">${C.kicker}</div>
    <div style="margin-top:52px;display:flex;flex-direction:column;gap:22px">
      ${C.funnel.map((f, i) => `
        <div class="frow row" style="gap:30px;opacity:0">
          <div class="fbar" style="width:0;height:62px;border-radius:6px;background:${i === 3 ? 'linear-gradient(90deg,#1a5fd6,#35c0f0)' : 'rgba(155,194,174,' + (0.34 - i * 0.07) + ')'}" data-w="${W[i]}"></div>
          <div class="num fnum" style="width:210px;text-align:right">0</div>
          <div class="lbl">${f[0]}</div>
        </div>`).join('')}
    </div>
    <div id="tasks" class="mono" style="margin-top:40px;font:600 30px var(--mono);color:var(--term-agent);opacity:0">across 13 tasks &middot; train 14,172 / val 724 / test 1,044</div>
    <div id="n" class="body2" style="margin-top:46px;max-width:1300px;opacity:0">${C.note}</div>
    <div id="l" class="mono" style="margin-top:22px;font:500 26px var(--mono);color:var(--term-finding);opacity:0">${C.leak}</div>
  </div>`;
export const build = (c) => {
  c.frame();
  c.add(c.$('#k'), { opacity: [0, 1], x: [-14, 0] }, c.t(150));
  c.$$('.frow').forEach((el, i) => {
    const at = c.t(600 + i * 620);
    const bar = el.querySelector('.fbar');
    c.add(el, { opacity: [0, 1], duration: 300 }, at);
    c.add(bar, { width: [0, Number(bar.dataset.w)], duration: 820 }, at);
    c.counter(el.querySelector('.fnum'), C.funnel[i][1], at, 900);
  });
  c.add(c.$('#tasks'), { opacity: [0, 1], duration: 620 }, c.t(3300));
  c.add(c.$('#n'), { opacity: [0, 1], duration: 700 }, c.t(4200));
  c.add(c.$('#l'), { opacity: [0, 1], duration: 700 }, c.t(5400));
};
