import { useEffect, useState } from 'react';
import { Icon, LogoMark } from '../components/Brand';
import { SITE_CONFIG } from '../lib/content';

export function Nav() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <header className={`nav${scrolled ? ' nav--scrolled' : ''}`}>
      <div className="nav__inner">
        <a className="nav__brand" href="#top" aria-label="UCSA — back to top">
          <LogoMark />
          <span>UCSA</span>
          <span className="nav__brand-tag">{SITE_CONFIG.release}</span>
        </a>
        <nav className="nav__links" aria-label="Primary">
          <a className="nav__link" href="#architecture">Architecture</a>
          <a className="nav__link" href="#research">Paper</a>
          <a className="nav__link" href="#docs">Docs</a>
          <a
            className="nav__link"
            href={SITE_CONFIG.repoUrl}
            target="_blank"
            rel="noreferrer noopener"
          >
            GitHub <Icon name="arrow" style={{ marginLeft: 4, verticalAlign: -2 }} />
          </a>
          <a className="nav__cta" href="#quickstart">
            Get started
          </a>
        </nav>
      </div>
    </header>
  );
}