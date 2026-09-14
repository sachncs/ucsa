import { LogoMark } from './Brand';
import { SITE_CONFIG } from '../lib/content';

export function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer__grid">
          <div className="footer__brand-col">
            <div className="footer__brand">
              <LogoMark />
              <span>UCSA</span>
            </div>
            <p className="footer__about">
              A research-grade foundation model whose computation orbits a
              single persistent differentiable cognitive state. Apache-2.0,
              open source, and built for the next decade of cognitive
              architectures.
            </p>
          </div>

          <div>
            <p className="footer__col-title">Project</p>
            <ul className="footer__list">
              <li>
                <a href={SITE_CONFIG.repoUrl} target="_blank" rel="noreferrer noopener">
                  Source on GitHub ↗
                </a>
              </li>
              <li>
                <a href={SITE_CONFIG.issuesUrl} target="_blank" rel="noreferrer noopener">
                  Issues ↗
                </a>
              </li>
              <li>
                <a href={SITE_CONFIG.discussionsUrl} target="_blank" rel="noreferrer noopener">
                  Discussions ↗
                </a>
              </li>
              <li>
                <a href={SITE_CONFIG.changelogUrl} target="_blank" rel="noreferrer noopener">
                  Changelog
                </a>
              </li>
            </ul>
          </div>

          <div>
            <p className="footer__col-title">Documentation</p>
            <ul className="footer__list">
              <li>
                <a href={`${SITE_CONFIG.docsUrl}/getting-started.md`} target="_blank" rel="noreferrer noopener">
                  Getting started
                </a>
              </li>
              <li>
                <a href={`${SITE_CONFIG.docsUrl}/architecture.md`} target="_blank" rel="noreferrer noopener">
                  Architecture
                </a>
              </li>
              <li>
                <a href={`${SITE_CONFIG.docsUrl}/api-reference.md`} target="_blank" rel="noreferrer noopener">
                  API reference
                </a>
              </li>
              <li>
                <a href={`${SITE_CONFIG.docsUrl}/tutorials.md`} target="_blank" rel="noreferrer noopener">
                  Tutorials
                </a>
              </li>
            </ul>
          </div>

          <div>
            <p className="footer__col-title">Research</p>
            <ul className="footer__list">
              <li>
                <a href={SITE_CONFIG.paperUrl} target="_blank" rel="noreferrer noopener">
                  Paper draft
                </a>
              </li>
              <li>
                <a href={SITE_CONFIG.paperTablesUrl} target="_blank" rel="noreferrer noopener">
                  Paper tables
                </a>
              </li>
              <li>
                <a href={SITE_CONFIG.contributingUrl} target="_blank" rel="noreferrer noopener">
                  Contributing
                </a>
              </li>
            </ul>
          </div>

          <div>
            <p className="footer__col-title">Community</p>
            <ul className="footer__list">
              <li>
                <a href="https://github.com/sachncs/ucsa/blob/master/CODE_OF_CONDUCT.md" target="_blank" rel="noreferrer noopener">
                  Code of conduct
                </a>
              </li>
              <li>
                <a href="https://github.com/sachncs/ucsa/blob/master/SECURITY.md" target="_blank" rel="noreferrer noopener">
                  Security
                </a>
              </li>
              <li>
                <a href="https://github.com/sachncs/ucsa/blob/master/SUPPORT.md" target="_blank" rel="noreferrer noopener">
                  Support
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="footer__bottom">
          <div className="footer__bottom-meta">
            <span>Apache-2.0 · 2026</span>
            <span className="footer__status">
              <span className="footer__status-dot" />
              <span>build passing</span>
            </span>
          </div>
          <div>
            <span>research platform · {SITE_CONFIG.release}.0</span>
          </div>
        </div>
      </div>
    </footer>
  );
}