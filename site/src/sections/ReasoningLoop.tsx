import { useReveal } from '../lib/useReveal';
import { TIMELINE_STEPS } from '../lib/content';
import { BASE_URL } from '../lib/base';

export function reasoning.Loop() {
  const ref = useReveal<HTMLDivElement>();
  return (
    <section className="section section--alt" id="reasoning">
      <div className="container">
        <div className="section__head reveal is-in">
          <span className="section__eyebrow eyebrow">how UCSA works</span>
          <h2 className="section__title">Eight steps. One computation.</h2>
          <p className="section__lede">
            The full forward pass, end to end, in the order the computation
            actually executes.
          </p>
        </div>

        <div className="timeline" ref={ref}>
          {TIMELINE_STEPS.map((step, i) => (
            <div key={step.title} className="step reveal is-in">
              <span className="step__num">
                {String(i + 1).padStart(2, '0')}
              </span>
              <h3 className="step__title">{step.title}</h3>
              <p className="step__desc">{step.desc}</p>
            </div>
          ))}
        </div>

        <div className="diagram-frame reveal is-in">
          <img
            src={`${BASE_URL}reasoning-loop.svg`}
            alt="Reasoning loop timeline: input flows into four iterations of the operator F producing logits, with JEPA chain back-edges"
          />
        </div>
      </div>
    </section>
  );
}