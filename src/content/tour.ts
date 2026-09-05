/** Guided tour: 5 steps, scrolls to each section, highlights nothing destructively. */

interface TourStep {
  sel: string;
  text: string;
}

const STEPS: TourStep[] = [
  {
    sel: '#claim',
    text: 'This is the single sentence the whole page argues for. Everything below is a way to test it yourself — the tour takes ~90 seconds.',
  },
  {
    sel: '#lab1',
    text: 'Lab 1: a sentence is streaming through a 96-neuron toy BDH cell right now. Left: neuron activity. Right: the synapse matrix — the ONLY memory. Gold rings are Hebbian writes happening this instant. Try γ → 0 to watch it forget.',
  },
  {
    sel: '#lab2',
    text: 'Lab 2: one causal linear-attention output, evaluated two ways. Green reads a fixed recurrent matrix; blue sums the explicit history. Their maximum component error should stay near floating-point noise. Orange softmax is intentionally separate and need not match.',
  },
  {
    sel: '#lab3',
    text: 'Lab 3: stress a separate generic associative memory. As random associations share one fixed matrix, wrong values can become closer competitors. The live curve is seed- and setup-dependent evidence of interference—not a measured BDH capacity law.',
  },
  {
    sel: '#bdh',
    text: 'Finally, the real thing: the exact BDH-GPU equations, what moves at inference (only activity and synaptic state — weights are frozen), and how BDH-CQ uses the same fabric to absorb ARC-AGI demonstrations without a chain of thought.',
  },
];

export function initTour(): void {
  const overlay = document.getElementById('tour-overlay')!;
  const card = document.getElementById('tour-card')!;
  const badge = document.getElementById('tour-step-badge')!;
  const text = document.getElementById('tour-text')!;
  const btn = document.getElementById('tour-btn') as HTMLButtonElement;
  const prev = document.getElementById('tour-prev') as HTMLButtonElement;
  const next = document.getElementById('tour-next') as HTMLButtonElement;
  const exit = document.getElementById('tour-exit') as HTMLButtonElement;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const focusableSelector = [
    'a[href]',
    'button:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
  ].join(',');
  let i = 0;
  let returnFocus: HTMLElement | null = null;
  let positionFrame = 0;

  overlay.setAttribute('aria-hidden', String(overlay.hidden));
  badge.setAttribute('aria-live', 'polite');
  card.setAttribute('tabindex', '-1');

  function focusableElements(): HTMLElement[] {
    return Array.from(card.querySelectorAll<HTMLElement>(focusableSelector))
      .filter((el) => el.getClientRects().length > 0);
  }

  function show(n: number): void {
    i = Math.max(0, Math.min(STEPS.length - 1, n));
    const s = STEPS[i];
    const el = document.querySelector(s.sel);
    if (el) el.scrollIntoView({ behavior: reducedMotion.matches ? 'auto' : 'smooth', block: 'start' });
    badge.textContent = `Step ${i + 1} of ${STEPS.length}`;
    text.textContent = s.text;
    prev.disabled = i === 0;
    next.textContent = i === STEPS.length - 1 ? 'Done' : 'Next';
  }

  function positionCard(): void {
    const el = document.querySelector(STEPS[i].sel);
    if (!el) return;
    const r = el.getBoundingClientRect();
    const viewportHeight = window.visualViewport?.height ?? window.innerHeight;
    const edge = 16;
    const gap = 12;
    const cardHeight = Math.min(card.scrollHeight, viewportHeight - edge * 2);
    const below = r.bottom + gap;
    const above = r.top - cardHeight - gap;
    const preferred = below + cardHeight <= viewportHeight - edge ? below : above;
    const maxTop = Math.max(edge, viewportHeight - cardHeight - edge);
    card.style.top = `${Math.min(maxTop, Math.max(edge, preferred))}px`;
  }

  function schedulePosition(): void {
    cancelAnimationFrame(positionFrame);
    positionFrame = requestAnimationFrame(positionCard);
  }

  function open(): void {
    returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : btn;
    overlay.hidden = false;
    overlay.setAttribute('aria-hidden', 'false');
    show(0);
    schedulePosition();
    requestAnimationFrame(() => next.focus());
  }
  function close(): void {
    if (overlay.hidden) return;
    overlay.hidden = true;
    overlay.setAttribute('aria-hidden', 'true');
    cancelAnimationFrame(positionFrame);
    const focusTarget = returnFocus?.isConnected ? returnFocus : btn;
    returnFocus = null;
    focusTarget.focus({ preventScroll: true });
  }

  btn.addEventListener('click', open);
  prev.addEventListener('click', () => { show(i - 1); schedulePosition(); });
  next.addEventListener('click', () => {
    if (i === STEPS.length - 1) close();
    else { show(i + 1); schedulePosition(); }
  });
  exit.addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
  window.addEventListener('resize', () => { if (!overlay.hidden) schedulePosition(); });
  window.addEventListener('scroll', () => { if (!overlay.hidden) schedulePosition(); }, { passive: true });
  window.visualViewport?.addEventListener('resize', () => { if (!overlay.hidden) schedulePosition(); });

  document.addEventListener('keydown', (e) => {
    if (overlay.hidden) return;
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
      return;
    }
    if (e.key === 'Tab') {
      const focusable = focusableElements();
      if (!focusable.length) {
        e.preventDefault();
        card.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && (active === first || !card.contains(active))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (active === last || !card.contains(active))) {
        e.preventDefault();
        first.focus();
      }
      return;
    }
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      next.click();
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      prev.click();
    }
  });
}
