import { useReveal } from '../lib/useReveal';
import { BANKS } from '../lib/content';

export function Banks() {
  const ref = useReveal<HTMLDivElement>();
  return (
    <section className="section" id="banks">
      <div className="container">
        <div className="section__head reveal is-in">
          <span className="section__eyebrow eyebrow">the seven banks</span>
          <h2 className="section__title">Anatomy of the PCS.</h2>
          <p className="section__lede">
            Six banks flow through the operator’s attention stream. The seventh
            — <code>intent</code> — is held out so the origination generator is
            the only path from intent to behaviour.
          </p>
        </div>

        <div className="banks" ref={ref}>
          {BANKS.map((b) => (
            <article
              key={b.name}
              className={`bank reveal is-in${b.full ? ' bank--full' : ''}`}
              style={{ ['--bank-color' as string]: b.color }}
            >
              <span className="bank__indicator" aria-hidden />
              <header className="bank__head">
                <span className="bank__name">{b.name}</span>
                <span className="bank__tokens">{b.tokens}</span>
              </header>
              <p className="bank__role">{b.role}</p>
              <div className="bank__meta">
                <div className="bank__meta-row">
                  <span className="bank__meta-key">role</span>
                  <span className="bank__meta-val">{b.meta.role}</span>
                </div>
                <div className="bank__meta-row">
                  <span className="bank__meta-key">path</span>
                  <span>
                    <span
                      className={`bank__pill ${
                        String(b.meta.path).startsWith('held')
                          ? 'bank__pill--held'
                          : 'bank__pill--stream'
                      }`}
                    >
                      {b.meta.path}
                    </span>
                  </span>
                </div>
                <div className="bank__meta-row">
                  <span className="bank__meta-key">shape</span>
                  <span className="bank__meta-val">{b.meta.tokens}</span>
                </div>
                {b.meta.service ? (
                  <div className="bank__meta-row">
                    <span className="bank__meta-key">service</span>
                    <span className="bank__meta-val">{b.meta.service}</span>
                  </div>
                ) : null}
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}