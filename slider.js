// slider.js

// Grab all the pieces
const slider    = document.getElementById('slides');
const prevBtn   = document.getElementById('prev');
const nextBtn   = document.getElementById('next');
const slides    = Array.from(document.querySelectorAll('.slide'));
let current     = 0;

// Apply the basic flex setup so each .slide sits side-by-side
slider.style.display    = 'flex';
slider.style.overflow   = 'hidden';            // hide the offscreen slides
slider.style.transition = 'transform 0.5s ease';

// Make each slide full width
slides.forEach(s => {
  s.style.flex = '0 0 100%';
});

// Helper to show the slide at index “i”
function show(i) {
  current = (i + slides.length) % slides.length;
  slider.style.transform = `translateX(-${current * 100}%)`;
}

// Wire up the buttons
prevBtn.addEventListener('click', () => show(current - 1));
nextBtn.addEventListener('click', () => show(current + 1));

// Kick it off at slide 0
show(0);
