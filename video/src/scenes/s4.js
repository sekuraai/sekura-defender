import { COPY } from '../content.js';
const C = COPY.s4;
export const html = () => `
  <div class="abs" style="left:140px;top:130px;width:1660px">
    <div id="k" class="kicker" style="opacity:0">${C.kicker}</div>
    <div class="row" style="margin-top:30px;gap:16px;flex-wrap:wrap">
      ${C.spec.map((s) => `<span class="pillbox sp" style="opacity:0;font-size:25px">${s}</span>`).join('')}
    </div>
    <div id="h" class="h2" style="margin-top:50px;font-size:46px;opacity:0">${C.head}</div>
    <div id="b" class="body2" style="margin-top:24px;max-width:1420px;font-size:27px;opacity:0">${C.body}</div>
    <div class="win" id="tbl" style="left:0;top:330px;width:1160px;height:372px;opacity:0;position:absolute">
      <div class="bar"><span style="margin:0">matched batch positions &middot; epoch 1 &rarr; epoch 2</span></div>
      <div class="body" style="padding:16px 26px;font-size:24px;white-space:normal">
        <div class="row" style="color:var(--term-dim);font-size:21px;padding:4px 0">
          <span style="width:90px">pos</span><span style="width:150px">n</span>
          <span style="width:200px">epoch 1</span><span style="width:200px">epoch 2</span><span>change</span>
        </div>
        ${C.pairs.map(([p, n, a, b]) => {
          const ch = ((b - a) / a * 100).toFixed(1);
          return `<div class="prow row" style="padding:7px 0;opacity:0">
            <span style="width:90px;color:var(--term-gray)">${p}</span>
            <span style="width:150px;color:var(--term-gray)">${n.toLocaleString('en-US')}</span>
            <span style="width:200px;color:#fff">${a.toFixed(4)}</span>
            <span style="width:200px;color:#fff">${b.toFixed(4)}</span>
            <span style="color:var(--signal);font-weight:700">${ch}%</span></div>`;
        }).join('')}
      </div>
    </div>
    <div id="v" class="mono" style="position:absolute;left:1240px;top:360px;width:430px;font:700 34px/1.35 var(--mono);color:var(--signal);opacity:0">${C.verdict}</div>
    <div id="cav" class="mono" style="position:absolute;left:1240px;top:510px;width:430px;font:400 22px/1.5 var(--mono);color:var(--term-dim);opacity:0">Epoch 2 revisits seen data, so this evidences a working gradient and a correct supervised mask &mdash; not held-out capability.</div>
  </div>`;
export const build = (c) => {
  c.frame();
  c.add(c.$('#k'), { opacity: [0, 1], x: [-14, 0] }, c.t(150));
  c.add(c.$$('.sp'), { opacity: [0, 1], y: [12, 0], duration: 460, delay: c.stagger(130) }, c.t(400));
  c.add(c.$('#h'), { opacity: [0, 1], y: [16, 0], duration: 700 }, c.t(1100));
  c.add(c.$('#b'), { opacity: [0, 1], duration: 700 }, c.t(1600));
  c.add(c.$('#tbl'), { opacity: [0, 1], y: [20, 0], duration: 620 }, c.t(2700));
  c.add(c.$$('.prow'), { opacity: [0, 1], x: [-14, 0], duration: 420, delay: c.stagger(240) }, c.t(3300));
  c.add(c.$('#v'), { opacity: [0, 1], scale: [0.94, 1], duration: 680 }, c.t(5000));
  c.add(c.$('#cav'), { opacity: [0, 1], duration: 700 }, c.t(6000));
};
