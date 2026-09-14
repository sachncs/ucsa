import { SITE_CONFIG } from '../lib/content';

export function Closing() {
  return (
    <section id="closing" className="section">
      <div className="container">
        <div className="closing">
          <div className="closing__inner">
            <span className="closing__eyebrow">closing note</span>
            <h2 className="closing__title">
              A flagship <em>research system.</em>
            </h2>
            <p className="closing__sub">
              A clear architectural thesis. A navigable documentation surface.
              A trustworthy release process. A research direction worth
              following.
            </p>
            <div className="closing__ctas">
              <a className="btn btn--primary" href="#quickstart">
                Get started →
              </a>
              <a
                className="btn btn--ghost btn--on-dark"
                href={SITE_CONFIG.paperUrl}
                target="_blank"
                rel="noreferrer noopener"
              >
                Read the paper
              </a>
              <a
                className="btn btn--ghost btn--on-dark"
                href={SITE_CONFIG.repoUrl}
                target="_blank"
                rel="noreferrer noopener"
              >
                Source on GitHub ↗
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}