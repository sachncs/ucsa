/* UCSA site — minimal interactivity */
(function () {
  // Copy-to-clipboard for terminal blocks
  document.addEventListener('click', function (e) {
    var btn = e.target.closest('.ucsa-terminal__copy');
    if (!btn) return;
    e.preventDefault();
    var body = btn.closest('.ucsa-terminal').querySelector('.ucsa-terminal__body');
    if (!body) return;
    var text = body.innerText;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = 'copied';
        setTimeout(function () { btn.textContent = 'copy'; }, 1400);
      });
    }
  });

  // Subtle fade-in for hero and section visuals (motion-safe)
  if (!window.matchMedia || !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('ucsa-in-view');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -10% 0px' });

    document.querySelectorAll('.ucsa-section').forEach(function (el) {
      observer.observe(el);
    });
  }
})();
