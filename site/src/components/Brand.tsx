import type { SVGProps } from 'react';

export function LogoMark(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 64 64" role="img" aria-label="UCSA" {...props}>
      <defs>
        <linearGradient id="lm-bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#0a0e1a" />
          <stop offset="100%" stopColor="#1f2a4a" />
        </linearGradient>
        <radialGradient id="lm-ring" cx="0.5" cy="0.5" r="0.5">
          <stop offset="60%" stopColor="#0a0e1a" stopOpacity="0" />
          <stop offset="100%" stopColor="#7c5cff" stopOpacity="0.22" />
        </radialGradient>
      </defs>
      <rect width="64" height="64" rx="14" fill="url(#lm-bg)" />
      <g transform="translate(32 32)">
        <circle cx="0" cy="0" r="26" fill="url(#lm-ring)" />
        <g fill="#c0b6ff">
          <circle cx="0" cy="-18" r="3" />
          <circle cx="16" cy="-9" r="3" />
          <circle cx="16" cy="9" r="3" />
          <circle cx="0" cy="18" r="3" />
          <circle cx="-16" cy="9" r="3" />
          <circle cx="-16" cy="-9" r="3" />
        </g>
        <circle cx="0" cy="22" r="4.2" fill="#ff6a6a" />
        <circle cx="0" cy="22" r="4.2" fill="none" stroke="#ffd2d2" strokeWidth="0.7" opacity="0.6" />
        <circle cx="0" cy="0" r="8" fill="#ffffff" />
        <circle cx="0" cy="0" r="8" fill="none" stroke="#7c5cff" strokeWidth="1.2" />
        <circle cx="0" cy="0" r="4" fill="#0a0e1a" />
        <g stroke="#c0b6ff" strokeWidth="0.7" opacity="0.6">
          <line x1="0" y1="-7" x2="0" y2="-15" />
          <line x1="6" y1="-5" x2="13" y2="-7" />
          <line x1="6" y1="5" x2="13" y2="7" />
          <line x1="0" y1="7" x2="0" y2="15" />
          <line x1="-6" y1="5" x2="-13" y2="7" />
          <line x1="-6" y1="-5" x2="-13" y2="-7" />
        </g>
        <line x1="0" y1="7" x2="0" y2="18" stroke="#ff6a6a" strokeWidth="0.7" strokeDasharray="1 1" opacity="0.7" />
      </g>
    </svg>
  );
}

type IconName =
  | 'check'
  | 'shield'
  | 'python'
  | 'license'
  | 'doc'
  | 'compare'
  | 'beaker'
  | 'config'
  | 'arrow'
  | 'github';

export function Icon({ name, ...props }: { name: IconName } & SVGProps<SVGSVGElement>) {
  const stroke = props.color ?? 'currentColor';
  const base = { fill: 'none', stroke, strokeWidth: 1.6, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
  switch (name) {
    case 'check':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" {...props}>
          <path d="M5 10l3 3 7-7" {...base} />
        </svg>
      );
    case 'shield':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" {...props}>
          <path d="M10 3l6 3v4c0 4-2.5 7-6 7s-6-3-6-7V6l6-3z" {...base} />
          <path d="M8 10l1.6 1.6L13 8" {...base} strokeWidth={1.4} />
        </svg>
      );
    case 'python':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" {...props}>
          <rect x="4" y="3" width="12" height="14" rx="2" {...base} />
          <path d="M7 8h6M7 11h6M7 14h4" {...base} strokeWidth={1.4} />
        </svg>
      );
    case 'license':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" {...props}>
          <circle cx="10" cy="10" r="7" {...base} />
          <path d="M7 10l2 2 4-4" {...base} strokeWidth={1.4} />
        </svg>
      );
    case 'doc':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" {...props}>
          <rect x="4" y="3" width="12" height="14" rx="2" {...base} />
          <path d="M7 8h6M7 11h6M7 14h4" {...base} strokeWidth={1.4} />
        </svg>
      );
    case 'compare':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" {...props}>
          <path d="M3 17l5-5 4 4 5-5" {...base} />
          <path d="M3 10h14" {...base} strokeWidth={1.4} />
        </svg>
      );
    case 'beaker':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" {...props}>
          <path d="M5 4h10v12H5z" {...base} />
          <path d="M7 8l2 2 4-4" {...base} strokeWidth={1.4} />
        </svg>
      );
    case 'config':
      return (
        <svg viewBox="0 0 20 20" width="18" height="18" {...props}>
          <rect x="3" y="4" width="14" height="12" rx="2" {...base} />
          <path d="M3 8h14" {...base} />
          <circle cx="6" cy="12" r="0.8" fill={stroke} stroke="none" />
          <circle cx="9" cy="12" r="0.8" fill={stroke} stroke="none" />
        </svg>
      );
    case 'arrow':
      return (
        <svg viewBox="0 0 20 20" width="14" height="14" {...props}>
          <path d="M4 10h12M11 5l5 5-5 5" {...base} strokeWidth={1.8} />
        </svg>
      );
    case 'github':
      return (
        <svg viewBox="0 0 20 20" width="14" height="14" {...props}>
          <path
            d="M10 1.5C5.3 1.5 1.5 5.3 1.5 10c0 3.8 2.5 7 5.8 8.1.4.1.6-.2.6-.4v-1.5c-2.4.5-2.9-1-2.9-1-.4-1-.9-1.3-.9-1.3-.8-.5.1-.5.1-.5.8.1 1.3.9 1.3.9.8 1.3 2 .9 2.5.7.1-.5.3-.9.5-1.1-1.9-.2-3.9-1-3.9-4.2 0-.9.3-1.7.9-2.3-.1-.2-.4-1.1.1-2.3 0 0 .7-.2 2.4.9.7-.2 1.5-.3 2.2-.3.7 0 1.5.1 2.2.3 1.7-1.1 2.4-.9 2.4-.9.5 1.2.2 2.1.1 2.3.6.6.9 1.4.9 2.3 0 3.2-2 4-3.9 4.2.3.3.6.8.6 1.7v2.5c0 .2.2.5.6.4 3.3-1.1 5.8-4.3 5.8-8.1 0-4.7-3.8-8.5-8.5-8.5z"
            fill={stroke}
            stroke="none"
          />
        </svg>
      );
  }
}