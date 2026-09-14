import { useReveal } from '../lib/useReveal';
import { METRICS } from '../lib/content';
import { Icon } from '../components/Brand';

export function Proof() {
  const ref = useReveal<HTMLDivElement>();
  return (
    <section className="section" id="proof">
      <div className="container">
        <div className="section__head reveal is-in">
          <span className="section__eyebrow eyebrow">proof</span>
          <h2 className="section__title">Evidence, not marketing.</h2>
          <p className="section__lede">
            The project earns trust by shipping reproducible tests, deterministic
            evals, and a working paper draft — not by claiming benchmark wins it
            has not measured.
          </p>
        </div>

        <div className="metrics" ref={ref}>
          {METRICS.map((m) => (
            <div key={m.title} className="metric reveal is-in">
              <span className="metric__icon">
                <Icon name={m.icon as never} />
              </span>
              <div className="metric__value">
                {m.value}
                {m.suffix ? (
                  <span className="metric__value-suffix">{m.suffix}</span>
                ) : null}
              </div>
              <h3 className="metric__title">{m.title}</h3>
              <p className="metric__desc">{m.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}