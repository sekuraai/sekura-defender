// Grab still frames for inspection. Usage: node tools/snap.mjs 1.5 4 9.2 ...  (seconds) → out/snaps/t<sec>.png
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { serve } from './serve.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const dir = join(ROOT, 'out/snaps'); mkdirSync(dir, { recursive: true });
const { server, url } = await serve();
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
const errors = []; page.on('pageerror', (e) => errors.push(e.message)); page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
await page.goto(`${url}/src/index.html`);
await page.waitForFunction(() => window.__ready === true, null, { timeout: 15000 }).catch(() => {});
if (errors.length) console.log('PAGE ERRORS:\n' + errors.join('\n'));
for (const s of process.argv.slice(2)) {
  await page.evaluate((ms) => window.__seek(ms), Math.round(Number(s) * 1000));
  const f = join(dir, `t${s}.png`); await page.screenshot({ path: f }); console.log(f);
}
await browser.close(); server.close();
