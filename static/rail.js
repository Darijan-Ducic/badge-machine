// The rail: the one switch between pages, top left (pre-coach app.html:14-19). Fills <nav class="rail"> and marks
// the current page. Include as <nav class="rail" aria-label="Pages"></nav><script src="rail.js"></script>.
// Icons are line art at pre-coach's .ic weight; mic, spark and file are pre-coach's own (cards.js:123, 132-133).
(() => {
  const PAGES = [
    ["index.html", "Record", '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/>'],
    ["preheat.html", "Preheat", '<path d="M12 3.5v4M12 16.5v4M3.5 12h4M16.5 12h4M6 6l2.6 2.6M15.4 15.4 18 18M6 18l2.6-2.6M15.4 8.6 18 6"/>'],
    ["clips.html", "Clips", '<rect x="3.5" y="5" width="17" height="14" rx="2.5"/><path d="M10 9.5v5l4-2.5z"/>'],
    ["stats.html", "Stats", '<path d="M5 20V12M12 20V5M19 20v-5"/>'],
    ["training.html", "Training", '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>'],
    ["pipeline.html", "Pipeline", '<rect x="3.5" y="3.5" width="7" height="6" rx="2"/><rect x="13.5" y="14.5" width="7" height="6" rx="2"/><path d="M7 9.5V15a2.5 2.5 0 0 0 2.5 2.5h4"/>'],
    ["track.html", "Tracker", '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="3"/>'],
  ];
  const here = location.pathname.split("/").pop() || "index.html";
  const rail = document.querySelector("nav.rail");
  for (const [href, label, icon] of PAGES) {
    const a = document.createElement("a");
    a.href = href;
    a.innerHTML = `<i class="ic"><svg viewBox="0 0 24 24" aria-hidden="true">${icon}</svg></i>${label}`;
    if (href === here) { a.className = "on"; a.setAttribute("aria-current", "page"); }
    rail.append(a);
  }
})();
