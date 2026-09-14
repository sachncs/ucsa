import { useState } from 'react';

type LineTone = 'comment' | 'prompt' | 'cmd' | 'accent' | 'output' | 'success' | 'blank';

type Line = { tone: LineTone; text: string };

const TERMINAL_LINES: Line[] = [
  { tone: 'comment', text: '# 1. install' },
  { tone: 'prompt', text: '$' },
  { tone: 'cmd', text: 'git clone https://github.com/sachncs/ucsa.git' },
  { tone: 'prompt', text: '$' },
  { tone: 'cmd', text: 'cd ucsa && python -m venv .venv && source .venv/bin/activate' },
  { tone: 'prompt', text: '$' },
  { tone: 'cmd', text: 'pip install -e ".[dev]"' },
  { tone: 'blank', text: '' },
  { tone: 'comment', text: '# 2. smoke test (5 steps, ~1 minute)' },
  { tone: 'prompt', text: '$' },
  { tone: 'cmd', text: '.venv/bin/python scripts/train.py \\' },
  { tone: 'accent', text: '    --max-steps 5 \\' },
  { tone: 'accent', text: '    --ckpt-every 0 --eval-every 0 \\' },
  { tone: 'accent', text: '    --skip-baselines --seed 42' },
  { tone: 'blank', text: '' },
  { tone: 'output', text: '  UCSA-small params: 63,000,000' },
  { tone: 'output', text: '  Device: mps' },
  { tone: 'output', text: '  Stack: JEPA=lewm(multi-step) + EMA=0.996 + recon(w=0.1) ...' },
  { tone: 'output', text: '  Final: loss=10.4123 val_ppl=33102 (best=33102@5)' },
  { tone: 'success', text: '  Wrote runs/ucsa-full-seed42.json' },
  { tone: 'blank', text: '' },
  { tone: 'comment', text: '# 3. full reproduction (~hours on GPU)' },
  { tone: 'prompt', text: '$' },
  { tone: 'cmd', text: '.venv/bin/python scripts/train.py --seed 42' },
  { tone: 'blank', text: '' },
  { tone: 'comment', text: '# 4. evaluate' },
  { tone: 'prompt', text: '$' },
  { tone: 'cmd', text: '.venv/bin/python scripts/eval.py \\' },
  { tone: 'accent', text: '    --ucsa-ckpt ckpts/ucsa-final.safetensors \\' },
  { tone: 'accent', text: '    --out-json runs/eval-ucsa-small.json' },
];

const QS_STEPS = [
  {
    title: 'Install',
    desc:
      'Clone, create a virtualenv, and install with the [dev] extra. Pulls in pytest, ruff, black, mypy, and pre-commit.',
  },
  {
    title: 'Smoke test',
    desc:
      'Five training steps, no checkpoints, no eval, no baseline comparison. Exits in under a minute. Confirms every subsystem wires up.',
  },
  {
    title: 'Full reproduction',
    desc:
      'Twelve thousand steps on fineweb-edu with the multi-step JEPA chain, hard-EMA target encoder, input-reconstruction head, and TC-JEPA conditioner all on.',
  },
  {
    title: 'Evaluate',
    desc:
      'Seed-deterministic rank-by-log-likelihood on HellaSwag, ARC, PIQA, and WinoGrande. Compare against the matched-compute baseline.',
  },
  {
    title: 'Probe',
    desc:
      'Inspect what each PCS bank has learned (top tokens, centroid cosine similarity) and where the intent bank is localising.',
  },
];

function copyText() {
  return TERMINAL_LINES.map((l) => {
    if (l.tone === 'blank') return '';
    if (l.tone === 'prompt') return '$';
    return l.text;
  }).join('\n');
}

function tone(t: LineTone): string {
  switch (t) {
    case 'comment':
      return 'terminal__comment';
    case 'prompt':
      return 'terminal__prompt';
    case 'cmd':
      return 'terminal__cmd';
    case 'accent':
      return 'terminal__accent';
    case 'success':
      return 'terminal__success';
    case 'output':
      return 'terminal__output';
    default:
      return '';
  }
}

export function Quickstart() {
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(copyText());
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* no-op */
    }
  };

  return (
    <section className="section section--alt" id="quickstart">
      <div className="container">
        <div className="section__head section__head--start reveal is-in">
          <span className="section__eyebrow eyebrow">quickstart</span>
          <h2 className="section__title">
            From clone to trained model in five minutes.
          </h2>
          <p className="section__lede">
            The smoke command exits in under a minute on CPU and proves the
            install wires up correctly. The full reproduction runs for hours on
            GPU.
          </p>
        </div>

        <div className="quickstart">
          <ol className="steps reveal is-in">
            {QS_STEPS.map((s) => (
              <li key={s.title}>
                <h3>{s.title}</h3>
                <p>{s.desc}</p>
              </li>
            ))}
          </ol>

          <div className="terminal reveal is-in">
            <div className="terminal__head">
              <span className="terminal__dots">
                <span className="terminal__dot terminal__dot--r" />
                <span className="terminal__dot terminal__dot--y" />
                <span className="terminal__dot terminal__dot--g" />
              </span>
              <span className="terminal__label">~/ucsa — zsh</span>
              <button className="terminal__copy" type="button" onClick={onCopy}>
                {copied ? 'copied' : 'copy'}
              </button>
            </div>
            <div className="terminal__body">
              {TERMINAL_LINES.map((line, i) => {
                if (line.tone === 'blank') {
                  return <div key={i}>{'\u00a0'}</div>;
                }
                if (line.tone === 'prompt') {
                  const next = TERMINAL_LINES[i + 1];
                  if (next && (next.tone === 'cmd' || next.tone === 'accent')) {
                    return (
                      <div key={i} className="terminal__line">
                        <span className="terminal__prompt">$</span>{' '}
                        <span className={tone(next.tone)}>{next.text}</span>
                      </div>
                    );
                  }
                  return (
                    <div key={i} className="terminal__line">
                      <span className="terminal__prompt">$</span>
                    </div>
                  );
                }
                if (line.tone === 'cmd' || line.tone === 'accent') {
                  const prev = TERMINAL_LINES[i - 1];
                  if (prev && prev.tone === 'prompt') {
                    return null;
                  }
                  return (
                    <div key={i} className="terminal__line">
                      <span className={tone(line.tone)}>{line.text}</span>
                    </div>
                  );
                }
                return (
                  <div key={i} className={tone(line.tone)}>
                    {line.text}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}