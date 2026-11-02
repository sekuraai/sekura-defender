import { COPY } from '../content.js';
const C = COPY.s8;
export const html = () => `
  <div class="abs" style="inset:0;background:radial-gradient(1400px 800px at 50% 45%, #121c1a 0%, #0B0F0E 72%)"></div>
  <div class="abs" style="left:0;right:0;top:400px;text-align:center">
    <div id="r" class="mono" style="font:700 56px var(--mono);background:var(--grad-text);-webkit-background-clip:text;background-clip:text;color:transparent;opacity:0">${C.repo}</div>
    <div id="rule8" style="width:0;height:2px;margin:34px auto 0;background:rgba(215,221,217,.2)"></div>
    <div id="l" class="mono" style="margin-top:32px;font:400 26px var(--mono);color:var(--term-dim);opacity:0">${C.line}</div>
    <div id="t" class="mono" style="margin-top:46px;font:500 24px var(--mono);color:var(--term-agent);opacity:0">Fireworks Training API &middot; 4&times;B300 &middot; 226 tests</div>
  </div>`;
export const build = (c) => {
  c.frame({ fadeIn: 500, fadeOut: 900 });
  c.add(c.$('#r'), { opacity: [0, 1], y: [18, 0], duration: 900 }, c.t(200));
  c.add(c.$('#rule8'), { width: [0, 520], duration: 800 }, c.t(900));
  c.add(c.$('#l'), { opacity: [0, 1], duration: 700 }, c.t(1400));
  c.add(c.$('#t'), { opacity: [0, 1], duration: 700 }, c.t(2000));
};
