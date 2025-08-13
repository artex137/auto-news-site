// slider.js — minimal, clickable slider with dots & autoplay
document.addEventListener("DOMContentLoaded", () => {
  const slider = document.querySelector(".slider-container, .slider");
  if (!slider) return;

  // Support both markup patterns:
  const track = slider.querySelector(".slides") || slider;
  let slides = Array.from(track.querySelectorAll(".slide"));
  if (slides.length === 0) {
    // nothing to do
    return;
  }

  // Build dots container if not present
  let dotsWrap = slider.querySelector(".dots");
  if (!dotsWrap) {
    dotsWrap = document.createElement("div");
    dotsWrap.className = "dots";
    slider.appendChild(dotsWrap);
  }

  let i = 0;
  const autoplayMs = 7000;
  let timer = null;

  // Ensure each slide is an <a> so clicking anywhere navigates
  slides.forEach((s) => {
    s.style.pointerEvents = "auto";
    s.querySelectorAll("*").forEach((el) => (el.style.pointerEvents = "auto"));
  });

  function renderDots() {
    dotsWrap.innerHTML = slides.map(() => `<span class="dot"></span>`).join("");
    dotsWrap.querySelectorAll(".dot").forEach((d, idx) =>
      d.addEventListener("click", () => go(idx))
    );
  }

  function go(n) {
    i = (n + slides.length) % slides.length;
    // Translate by percentage if using a .slides track
    if (track.classList.contains("slides")) {
      track.style.display = "flex";
      track.style.transition = "transform .6s cubic-bezier(.22,.61,.36,1)";
      track.style.transform = `translateX(${-i * 100}%)`;
      slides.forEach((s) => (s.style.minWidth = "100%"));
    } else {
      // Fallback: show/hide slides
      slides.forEach((s, k) => (s.style.display = k === i ? "block" : "none"));
    }
    // Activate dot
    dotsWrap.querySelectorAll(".dot").forEach((d, k) =>
      d.classList.toggle("active", k === i)
    );
  }

  function start() {
    if (autoplayMs > 0) timer = setInterval(() => go(i + 1), autoplayMs);
  }
  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
  }

  // Optional prev/next buttons if present
  const prev = slider.querySelector(".prev");
  const next = slider.querySelector(".next");
  if (prev) prev.addEventListener("click", () => go(i - 1));
  if (next) next.addEventListener("click", () => go(i + 1));

  // Pause on hover (desktop)
  slider.addEventListener("mouseenter", stop);
  slider.addEventListener("mouseleave", start);

  renderDots();
  go(0);
  start();
});
