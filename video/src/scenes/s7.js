import { COPY } from '../content.js';
const C = COPY.s7;
export const html = () => `
  <div class="abs" style="left:140px;top:150px;width:1640px">
    <div id="k" class="kicker" style="opacity:0;color:var(--term-finding)">${C.kicker}</div>
    <div class="win" id="gw" style="position:absolute;left:0;top:92px;width:1180px;height:330px;opacity:0">
      <div class="bar"><span style="margin:0">eval/gate.py &mdash; ship thresholds</span>
        <span id="gv" style="margin-left:auto;font-weight:800;color:var(--term-exploit);opacity:0">GATE: FAIL</span></div>
      <div class="body" style="padding:22px 28px;font-size:25px;white-space:normal">
        ${C.gate.map((g) => `<div class="grow row" style="padding:9px 0;opacity:0">
          <span style="width:520px;color:var(--term-gray)">${g[0]}</span>
          <span style="width:150px;color:#fff;font-weight:700">${g[1]}</span>
          <span style="color:var(--term-dim)">violates ${g[2]}</span></div>`).join('')}
      </div>
    </div>
    <div id="reg" class="abs" style="left:1240px;top:92px;width:420px;padding:28px 30px;border-radius:14px;background:rgba(255,122,122,.08);border:1px solid rgba(255,122,122,.3);opacity:0">
      <div class="mono" style="font:700 25px var(--mono);color:var(--term-exploit)">regression</div>
      <div class="mono" style="margin-top:16px;font:400 23px/1.5 var(--mono);color:var(--term-fg)">${C.reg}</div>
    </div>
    <div id="f" class="h2" style="position:absolute;left:0;top:500px;font-size:44px;max-width:1500px;opacity:0">${C.foot}</div>
    <div id="n" class="mono" style="position:absolute;left:0;top:618px;width:1500px;font:400 24px/1.5 var(--mono);color:var(--term-dim);opacity:0">
      At n=20 the dangerous-error rate is a single example &mdash; too thin to size the effect, and not a reason to discount it.
      A 250-row held-out triage set is the next thing to build, ahead of any RL.
    </div>
  </div>`;
export const build = (c) => {
  c.frame();
  c.add(c.$('#k'), { opacity: [0, 1], x: [-14, 0] }, c.t(150));
  c.add(c.$('#gw'), { opacity: [0, 1], y: [18, 0], duration: 620 }, c.t(450));
  c.add(c.$$('.grow'), { opacity: [0, 1], x: [-12, 0], duration: 420, delay: c.stagger(330) }, c.t(1000));
  c.add(c.$('#gv'), { opacity: [0, 1], scale: [0.85, 1], duration: 440 }, c.t(2300));
  c.add(c.$('#reg'), { opacity: [0, 1], x: [18, 0], duration: 620 }, c.t(3000));
  c.add(c.$('#f'), { opacity: [0, 1], y: [14, 0], duration: 700 }, c.t(4600));
  c.add(c.$('#n'), { opacity: [0, 1], duration: 700 }, c.t(5600));
};
