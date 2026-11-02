import { COPY } from '../content.js';
const C = COPY.s0;
export const html = () => `
  <div class="abs" style="inset:0;background:radial-gradient(1400px 800px at 50% 45%, #121c1a 0%, #0B0F0E 72%)"></div>
  <svg class="abs" width="1920" height="1080" style="left:0;top:0">
    <g id="grid0" opacity="0">
      ${Array.from({ length: 13 }, (_, i) => `<line x1="0" y1="${i * 90}" x2="1920" y2="${i * 90}" stroke="rgba(215,221,217,.05)" stroke-width="1"/>`).join('')}
      ${Array.from({ length: 22 }, (_, i) => `<line x1="${i * 90}" y1="0" x2="${i * 90}" y2="1080" stroke="rgba(215,221,217,.05)" stroke-width="1"/>`).join('')}
    </g>
  </svg>
  <div class="abs" style="left:0;right:0;top:382px;text-align:center">
    <div id="nm" class="h1" style="font-size:104px;opacity:0">${C.name}</div>
    <div id="rule" style="width:0;height:3px;margin:38px auto 0;background:linear-gradient(90deg,#1a5fd6,#35c0f0)"></div>
    <div id="sb" class="mono" style="margin-top:36px;font:500 32px var(--mono);color:var(--term-agent);opacity:0">${C.sub}</div>
    <div id="mt" class="mono" style="margin-top:18px;font:400 25px var(--mono);color:var(--term-dim);opacity:0">${C.meta}</div>
  </div>`;
export const build = (c) => {
  c.frame({ fadeIn: 1, fadeOut: 500 });
  c.add(c.$('#grid0'), { opacity: [0, 1], duration: 1400, ease: 'linear' }, c.t(0));
  c.add(c.$('#nm'), { opacity: [0, 1], y: [26, 0], duration: 1100 }, c.t(250));
  c.add(c.$('#rule'), { width: [0, 460], duration: 900 }, c.t(1100));
  c.add(c.$('#sb'), { opacity: [0, 1], y: [14, 0], duration: 800 }, c.t(1650));
  c.add(c.$('#mt'), { opacity: [0, 1], duration: 700 }, c.t(2200));
};
