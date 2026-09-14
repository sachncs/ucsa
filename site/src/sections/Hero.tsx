import { useReveal } from '../lib/useReveal';
import { Icon, } from '../components/Brand';
import { SITE_CONFIG } from '../lib/content';
import { STRIP } from '../lib/content';
import { BASE_URL } from '../lib/base';

export function Hero() {
  const ref = useReveal<HTMLDivElement>();
  return (
    <>
      <section className="hero" id="top">
        <div className="hero__bg" aria-hidden />
        <div className="container">
          <div className="hero__inner">
            <div className="hero__copy">
              <span className="hero__eyebrow">
                <span className="hero__eyebrow-dot" />
                research platform · {SITE_CONFIG.release}
              </span>
              <h1 className="hero__title">
                A foundation model built around one{' '}
                <em>persistent cognitive state.</em>
              </h1>
              <p className="hero__sub">
                <strong>UCSA</strong> is a research-grade architecture whose entire
                computation — language logits, JEPA predictions, memory, planning,
                tool — reads from or writes to the same seven-bank differentiable
                state. A Transformer is one realisation of the operator; the state
                is the thesis.
              </p>
              <div className="hero__ctas">
                <a className="btn btn--primary" href="#quickstart">
                  Get started <Icon name="arrow" className="arrow" />
                </a>
                <a className="btn btn--ghost btn--on-dark" href="#architecture">
                  Architecture
                </a>
                <a
                  className="btn btn--ghost btn--on-dark"
                  href={SITE_CONFIG.paperUrl}
                  target="_blank"
                  rel="noreferrer noopener"
                >
                  Read the paper
                </a>
              </div>
              <div className="hero__signals">
                <div className="hero__signal">
                  <span className="hero__signal-value">607</span>
                  <span className="hero__signal-label">tests passing</span>
                </div>
                <div className="hero__signal">
                  <span className="hero__signal-value">80%</span>
                  <span className="hero__signal-label">coverage gate enforced</span>
                </div>
                <div className="hero__signal">
                  <span className="hero__signal-value">3.11 · 3.12</span>
                  <span className="hero__signal-label">Python supported</span>
                </div>
                <div className="hero__signal">
                  <span className="hero__signal-value">Apache-2.0</span>
                  <span className="hero__signal-label">open source license</span>
                </div>
              </div>
            </div>
            <div className="hero__visual">
              <div className="hero__card">
                <div className="hero__card-head">
                  <span className="hero__card-title">persistent cognitive state</span>
                  <span className="hero__card-eq">Cₜ₊₁ = F(Cₜ, Oₜ)</span>
                </div>
                <img src={`${BASE_URL}pcs-hero.svg`} alt="Seven banks feeding a single transition operator F, JEPA multi-step chain, and the held-out intent bank" />
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="strip">
        <div className="container">
          <div className="strip__row" ref={ref}>
            {STRIP.map((item, i) => (
              <span key={item.label} className="strip__item">
                <span className="strip__dot" />
                <strong>{item.value}</strong>
                <span>{item.label}</span>
                {i < STRIP.length - 1 ? null : null}
              </span>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}