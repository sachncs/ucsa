import { useReveal } from '../lib/useReveal';
import { ARCH_CARDS } from '../lib/content';
import { BASE_URL } from '../lib/base';

export function Architecture() {
  const ref = useReveal<HTMLDivElement>();
  return (
    <section className="section section--alt" id="architecture">
      <div className="container">
        <div className="section__head reveal is-in">
          <span className="section__eyebrow eyebrow">architecture overview</span>
          <h2 className="section__title">A structured map, not prose.</h2>
          <p className="section__lede">
            Seven subsystems, one shared state. Each subsystem has a single
            responsibility and a single integration point: the PCS.
          </p>
        </div>

        <div className="arch" ref={ref}>
          {ARCH_CARDS.map((c) => (
            <article
              key={c.n}
              className={`arch__card arch__card--${c.span} reveal is-in`}
            >
              <span className="arch__num">{c.n}</span>
              <h3 className="arch__name">{c.name}</h3>
              <p className="arch__desc">{c.desc}</p>
              <span className="arch__tag">{c.tag}</span>
            </article>
          ))}

          <div className="arch__card arch__card--full arch__diagram reveal is-in">
            <img src={`${BASE_URL}architecture.svg`} alt="Architecture overview: PCS at center, surrounded by perception, reasoning loop, memory service, projection heads, auxiliary losses, eval harness" />
          </div>
        </div>
      </div>
    </section>
  );
}