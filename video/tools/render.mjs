// Frame-accurate render: seek the anime.js timeline per frame in headless Chromium, pipe PNGs to ffmpeg.
// Frames are a pure function of time, so the range is split across parallel workers and concatenated.
// Usage: node tools/render.mjs [--from 0] [--to 155] [--workers 4] [--out out/sekura-defender-submission.mp4]
import { chromium } from 'playwright';
import { spawn, execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { serve } from './serve.mjs';
import { FPS, WIDTH, HEIGHT, DURATION } from '../src/content.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > -1 ? process.argv[i + 1] : d; };
const from = Math.round(Number(arg('from', 0)) * FPS);
const to = Math.min(Math.round(Number(arg('to', DURATION / 1000)) * FPS), Math.round((DURATION / 1000) * FPS));
const WORKERS = Number(arg('workers', 4));
const OUT = resolve(ROOT, arg('out', 'out/sekura-defender-submission.mp4'));
const TMP = join(ROOT, 'out/.segments'); rmSync(TMP, { recursive: true, force: true }); mkdirSync(TMP, { recursive: true });

const { server, url } = await serve();
const browser = await chromium.launch({ args: ['--force-color-profile=srgb', '--disable-lcd-text', '--font-render-hinting=none'] });
const total = to - from; let done = 0; const t0 = Date.now();

async function worker(w, a, b) {
  const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
  const errors = []; page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${url}/src/index.html`);
  await page.waitForFunction(() => window.__ready === true, null, { timeout: 30000 });
  if (errors.length) throw new Error(errors.join('\n'));
  if (w === 0) writeFileSync(join(ROOT, 'audio/cues.json'), JSON.stringify(await page.evaluate(() => window.__cues), null, 1));
  const cdp = await page.context().newCDPSession(page);
  const seg = join(TMP, `seg${String(w).padStart(2, '0')}.mp4`);
  const ff = spawn('ffmpeg', ['-y', '-v', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'png', '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p', '-r', String(FPS), '-movflags', '+faststart', seg],
  { stdio: ['pipe', 'inherit', 'inherit'] });
  const closed = new Promise((ok, ko) => ff.on('close', (c) => (c ? ko(new Error(`ffmpeg ${c}`)) : ok())));
  for (let f = a; f < b; f++) {
    await page.evaluate((ms) => window.__seek(ms), (f * 1000) / FPS);
    const { data } = await cdp.send('Page.captureScreenshot', { format: 'png', optimizeForSpeed: true });
    if (!ff.stdin.write(Buffer.from(data, 'base64'))) await new Promise((ok) => ff.stdin.once('drain', ok));
    if (++done % 150 === 0) {
      const el = (Date.now() - t0) / 1000;
      process.stdout.write(`\r${done}/${total} frames · ${(done / el).toFixed(1)} fps · eta ${Math.round((total - done) / (done / el))}s   `);
    }
  }
  ff.stdin.end(); await closed; await page.close();
  return seg;
}

const per = Math.ceil(total / WORKERS);
const segs = await Promise.all(Array.from({ length: WORKERS }, (_, w) => {
  const a = from + w * per, b = Math.min(to, a + per);
  return a < b ? worker(w, a, b) : null;
}));
await browser.close(); server.close();

const list = join(TMP, 'list.txt');
writeFileSync(list, segs.filter(Boolean).map((s) => `file '${s}'`).join('\n'));
mkdirSync(dirname(OUT), { recursive: true });
execFileSync('ffmpeg', ['-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', '-movflags', '+faststart', OUT]);
rmSync(TMP, { recursive: true, force: true });
console.log(`\n${OUT}  (${total} frames in ${((Date.now() - t0) / 1000).toFixed(0)}s)`);
