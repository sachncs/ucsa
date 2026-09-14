export function Limitations() {
  return (
    <section className="section" id="limitations">
      <div className="container">
        <div className="section__head reveal is-in">
          <span className="section__eyebrow eyebrow">limitations & honesty</span>
          <h2 className="section__title">What the architecture does not claim.</h2>
          <p className="section__lede">
            This section is deliberately visible. Trust comes from stating the
            limits clearly, not from omitting them.
          </p>
        </div>

        <div className="callouts">
          <div className="callout callout--warn reveal is-in">
            <div className="callout__head">
              <span className="callout__icon">!</span>
              <span className="callout__title">scale-bound negative result</span>
            </div>
            <div className="callout__body">
              <p>
                At the 65M-on-fineweb scale the matched-compute perplexity gap
                between UCSA and the vanilla Transformer is real (+0.0197) but
                inside one pooled seed sd (2.14 sd away from zero, not 0 sd).
                The paper says so. Closing this requires running the full
                12k-step recipe at larger scale; that is in progress and not yet
                a positive result.
              </p>
            </div>
          </div>

          <div className="callout callout--accent reveal is-in">
            <div className="callout__head">
              <span className="callout__icon">→</span>
              <span className="callout__title">what needs validation</span>
            </div>
            <div className="callout__body">
              <ul className="callout__list">
                <li>
                  <strong>Origination is localisable at the small scale.</strong>
                  The headline claim (2–4 of 16 intent slots carry gradient,
                  ablation of an attributed slot moves the action, ablation of
                  an unattributed slot does not) has been measured on a
                  64-hidden 4-layer model trained on a copy task. Generalising
                  to the full-scale UCSA requires re-running the same probe.
                </li>
                <li>
                  <strong>Intent descent does not yet beat matched controls.</strong>
                  At the small scale, descent helps when the critic and the
                  realised outcome agree, and it hurts when they disagree. A
                  learned verifier trained on a genuine outcome signal is
                  required for the descent to be useful at scale.
                </li>
                <li>
                  <strong>Memory service runs in-process.</strong>
                  For paper-grade runs, the verification, consolidation, and
                  pruning workers should be separated into a separate process;
                  today they run on a background thread inside the trainer.
                </li>
              </ul>
            </div>
          </div>

          <div className="callout callout--good reveal is-in">
            <div className="callout__head">
              <span className="callout__icon">✓</span>
              <span className="callout__title">what is stable</span>
            </div>
            <div className="callout__body">
              <ul className="callout__list">
                <li>
                  <strong>Architecture.</strong> The seven-bank PCS, the
                  reasoning loop with differentiable carry, the multi-step JEPA
                  chain with EMA targets, and the held-out intent bank are
                  first-class.
                </li>
                <li>
                  <strong>Reproducibility.</strong> Deterministic seeding,
                  Hydra/OmegaConf config composition, seed-deterministic eval
                  harness, safetensors checkpoints with full metadata, and a CI
                  matrix that fails on coverage regression.
                </li>
                <li>
                  <strong>Protocol.</strong> Matched-compute baselines and a
                  standard LM eval harness catch negative results and let the
                  paper say so.
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}