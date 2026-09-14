import { ROLES, DOCS } from '../lib/content';

export function Docs() {
  return (
    <section className="section" id="docs">
      <div className="container">
        <div className="section__head reveal is-in">
          <span className="section__eyebrow eyebrow">documentation</span>
          <h2 className="section__title">Pick your entry point.</h2>
          <p className="section__lede">
            The docs are organised by role. Implementers go straight to the API
            reference; researchers start at the architecture doc and the paper;
            evaluators jump to the eval harness; contributors go through the dev
            setup first.
          </p>
        </div>

        <div className="roles">
          {ROLES.map((r) => (
            <article key={r.code} className="role reveal is-in">
              <span className="role__icon">{r.code}</span>
              <h3 className="role__title">{r.title}</h3>
              <p className="role__desc">{r.desc}</p>
              <div className="role__links">
                {r.links.map((l) => (
                  <a
                    key={l.label}
                    className="role__link"
                    href={l.href}
                    target={l.external ? '_blank' : undefined}
                    rel={l.external ? 'noreferrer noopener' : undefined}
                  >
                    → {l.label}
                  </a>
                ))}
              </div>
            </article>
          ))}
        </div>

        <div className="docs-grid">
          {DOCS.map((d) => (
            <a
              key={d.title}
              className="doc reveal is-in"
              href={d.href}
              target="_blank"
              rel="noreferrer noopener"
            >
              <span className="doc__eyebrow">{d.eyebrow}</span>
              <h4 className="doc__title">{d.title}</h4>
              <p className="doc__desc">{d.desc}</p>
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}