import { useReveal } from '../lib/useReveal';

export function Concept() {
  const ref = useReveal<HTMLDivElement>();
  return (
    <section className="section" id="concept">
      <div className="container">
        <div className="section__head section__head--start reveal is-in">
          <span className="section__eyebrow eyebrow">core concept</span>
          <h2 className="section__title">
            One state. Seven banks. <br />
            One chain.
          </h2>
          <p className="section__lede">
            UCSA replaces the conventional “stack more layers, hope it generalises”
            pattern with a single, persistent, differentiable state that every
            projection in the model reads from or writes to.
          </p>
        </div>

        <div className="concept" ref={ref}>
          <div className="concept__copy reveal is-in">
            <p>
              The Persistent Cognitive State (PCS) is a stack of seven learnable
              tensors — one per bank — all of shape{' '}
              <code>(num_tokens, hidden_size)</code>. Every projection in the
              model reads from or writes to this state; no other structure
              stores knowledge.
            </p>
            <p>
              The first six banks — <code>working</code>, <code>long_term</code>,
              <code> goal</code>, <code>episode</code>, <code>task</code>,
              <code> memory_index</code> — flow through the operator’s attention
              stream. The seventh, <code>intent</code>, is held out of the
              stream on purpose: it is the origination signal, and making the
              origination generator the <em>only</em> path from intent to
              behaviour is what makes per-slot attribution well posed.
            </p>
            <p>
              A forward pass runs the operator <code>N</code> times (default 4),
              each time writing a new PCS. A multi-step JEPA prediction chain
              then pairs consecutive working latents into{' '}
              <em>(predicted, target)</em> pairs and adds them to the loss.
            </p>
            <ul className="concept__list">
              <li>
                <span className="concept__check">✓</span>
                Everything is anchored to one state.
              </li>
              <li>
                <span className="concept__check">✓</span>
                Memory, goals, and intent are first-class tensors.
              </li>
              <li>
                <span className="concept__check">✓</span>
                The operator is interchangeable; the state is not.
              </li>
            </ul>
          </div>

          <aside className="equation reveal is-in">
            <p className="equation__eyebrow">the central equation</p>
            <p className="equation__formula">
              C<sub>t+1</sub> = <em>F</em>(C<sub>t</sub>, O<sub>t</sub>)
            </p>
            <p className="equation__note">
              A single transition operator updates one persistent state. Mamba,
              RWKV, and Hyena are alternative realisations of the same{' '}
              <em>F</em>; the rest of the system is unchanged.
            </p>
          </aside>
        </div>
      </div>
    </section>
  );
}