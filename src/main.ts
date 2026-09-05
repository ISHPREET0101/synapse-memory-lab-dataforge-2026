import { initLab1 } from './labs/lab1.js';
import { initLab2 } from './labs/lab2.js';
import { initLab3 } from './labs/lab3.js';
import { initTour } from './content/tour.js';
import './styles/main.css';

const start = (): void => {
  const lab1 = initLab1();
  initLab2();
  initLab3();
  initTour();

  // Pause the live simulation when the tab is hidden or Lab 1 is far off-screen
  // (keeps feedback instant and CPU polite).
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) lab1.setRunning(false);
    else lab1.setRunning(true);
  });

  const lab1Section = document.getElementById('lab1');
  if (lab1Section && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) lab1.setRunning(e.isIntersecting && !document.hidden);
      },
      { threshold: 0.05 },
    );
    io.observe(lab1Section);
  }

  if (import.meta.env?.DEV) {
    console.info('Synapse Memory Lab — dev mode. All computation is local.');
  }
};

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
else start();
