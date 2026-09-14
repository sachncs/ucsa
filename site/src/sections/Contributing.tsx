import { CONTRIB_STEPS, SITE_CONFIG } from '../lib/content';

export function Contributing() {
  return (
    <section className="section section--alt" id="contributing">
      <div className="container">
        <div className="section__head reveal is-in">
          <span className="section__eyebrow eyebrow">contributing</span>
          <h2 className="section__title">A clean developer onboarding path.</h2>
          <p className="section__lede">
            The contributor workflow is short, structured, and welcoming. Open
            an issue, send a PR, watch the CI run.
          </p>
        </div>

        <div className="contrib">
          <div className="contrib__copy reveal is-in">
            <p>
              UCSA is research code with a software-quality bar. The contributor
              workflow mirrors that: dev setup, lint, type check, tests, PR
              template, and a code review checklist. The maintainers answer
              issues on a best-effort basis.
            </p>
            <p>
              For substantive changes — new banks, new loss terms, operator
              swaps — open an issue first so the design discussion happens
              before the code review. For fixes and small improvements, a PR
              with tests and a docs update is enough.
            </p>
            <p>
              The full guide lives at{' '}
              <a
                href={SITE_CONFIG.contributingUrl}
                target="_blank"
                rel="noreferrer noopener"
              >
                CONTRIBUTING.md
              </a>
              . It covers every command a contributor will need, the
              conventional-commits style, and the merge criteria.
            </p>
            <div className="contrib__ctas">
              <a
                className="btn btn--primary"
                href={SITE_CONFIG.contributingUrl}
                target="_blank"
                rel="noreferrer noopener"
              >
                Read the contributor guide →
              </a>
              <a
                className="btn btn--ghost"
                href={SITE_CONFIG.issuesUrl}
                target="_blank"
                rel="noreferrer noopener"
              >
                Open an issue ↗
              </a>
            </div>
          </div>

          <div className="contrib__flow">
            {CONTRIB_STEPS.map((s) => (
              <div key={s.num} className="contrib-step reveal is-in">
                <span className="contrib-step__num">step {s.num}</span>
                <h3 className="contrib-step__title">{s.title}</h3>
                <p className="contrib-step__desc">{s.desc}</p>
                <code className="contrib-step__cmd">{s.cmd}</code>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}