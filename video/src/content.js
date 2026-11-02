// Single source of truth for the Sekura Defender submission video.
// Every figure here is copied from data/fireworks/report-{base,tuned}.json and the run logs.
export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;

export const SCENES = [
  { id: 's0', title: 'Title',     start: 0,     dur: 6000 },
  { id: 's1', title: 'The task',  start: 6000,  dur: 13000 },
  { id: 's2', title: 'Validators',start: 19000, dur: 13000 },
  { id: 's3', title: 'Data',      start: 32000, dur: 13000 },
  { id: 's4', title: 'Training',  start: 45000, dur: 13000 },
  { id: 's5', title: 'Results',   start: 58000, dur: 20000 },
  { id: 's6', title: 'Planner',   start: 78000, dur: 11000 },
  { id: 's7', title: 'Gate',      start: 89000, dur: 11000 },
  { id: 's8', title: 'Close',     start: 100000, dur: 6000 },
];
export const DURATION = 106000;

// Per-task results. base = session sampler (same serving path as tuned).
export const TASKS = [
  ['attacktree_repair', 0.083, 0.817],
  ['triage',            0.129, 0.591],
  ['planner',           0.600, 1.000],
  ['mapping',           0.208, 0.551],
  ['attacktree',        0.255, 0.594],
  ['remediation',       0.303, 0.407],
  ['threat_model',      0.827, 0.887],
  ['cvss',              0.350, 0.400],
  ['grounded_qa',       0.809, 0.841],
  ['detection',         0.839, 0.869],
  ['safety',            0.806, 0.808],
  ['classify_cwe',      0.600, 0.600],
  ['classify_category', 0.450, 0.350],
];
export const OVERALL = { base: 0.4690, tuned: 0.6653 };

export const COPY = {
  s0: {
    name: 'Sekura Defender',
    sub: 'rank-32 LoRA · qwen3p6-35b-a3b · Fireworks Training API',
    meta: 'defensive security · 13 tasks · 238 held-out examples',
  },
  s1: {
    kicker: 'The task',
    head: 'A security scanner that pays frontier prices for schema work.',
    body: 'Sekura runs a multi-agent scanner. Its 370,231-record knowledge corpus was only ever used as retrieval material, so every scan paid frontier latency and cost for work that is narrow and repetitive.',
    jobs: ['author attack trees', 'write detections', 'plan recon coverage', 'triage findings', 'map CWE → ATT&CK', 'compute CVSS'],
  },
  s2: {
    kicker: 'Why fine-tune, not prompt',
    head: 'These tasks have deterministic validators the base model never trained against.',
    items: [
      ['attack-tree schema', '8 invariant classes, exact labels'],
      ['CVSS v3.1 scorer', 'arithmetic, not opinion'],
      ['recon coverage invariant', 'empty plan against a non-empty inventory = 0'],
    ],
    foot: 'So the training signal and the benchmark are objective, not judged.',
  },
  s3: {
    kicker: 'The data',
    funnel: [['raw records', 370231], ['curated', 160080], ['grounded bundles', 16503], ['training examples', 15940]],
    note: 'Splits key on entity — CVE, source path, subtree id, rule-set version — so regenerating the dataset cannot move an example across the boundary.',
    leak: '3,612 intel rows hold only 885 unique findings. Row-level splitting would have leaked.',
  },
  s4: {
    kicker: 'Training',
    spec: ['94 steps · 2 epochs', '3,660 balanced examples', '4×B300 · 15.3 min', 'rank 32 · lr 1e-4 cosine'],
    head: 'Per-step loss is not evidence here.',
    body: 'Batches are length-bucketed, so consecutive steps differ in task and difficulty. The series wandered 0.37–1.52 with no trend while the model trained well. Matched positions across epochs are comparable:',
    pairs: [[1, 1401, 1.2458, 0.4705], [2, 263, 1.0783, 0.5139], [3, 89, 0.9431, 0.3739], [4, 43, 1.3296, 0.9307], [5, 40, 1.5205, 1.0142]],
    verdict: 'improved on 5/5 matched batches · mean −47.6%',
  },
  s5: { kicker: 'Before and after', head: '238 held-out examples · 20 per task · greedy · one serving path' },
  s6: {
    kicker: 'The result that matters',
    body: 'Sekura had a production bug: the planner called a site static, emitted an empty plan, and the scan returned nothing.',
    foot: 'On coverage rule sets it never trained on — the difference between a planner and a lookup table.',
  },
  s7: {
    kicker: 'What we are not claiming',
    gate: [['triage.dangerous_rate', '0.05', '≤ 0.03'], ['attacktree_repair.exact', '0.45', '≥ 0.60'], ['cvss.exact', '0.40', '≥ 0.90']],
    reg: 'dangerous_rate is a regression: base 0.000 → tuned 0.05. Accepting a finding that should be rejected is the error that reaches a customer.',
    foot: 'Training worked. This checkpoint is not shippable yet. Both are true.',
  },
  s8: { repo: 'github.com/sekuraai/sekura-defender', line: 'Pipeline, eval harness and raw reports. Corpus withheld — licensed material.' },
};
