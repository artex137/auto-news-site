// slider.js

const sliderSection    = document.getElementById('slider');
const slidesContainer  = document.getElementById('slides');
const slides           = document.querySelectorAll('.slide');
const prevBtn          = document.getElementById('prev');
const nextBtn          = document.getElementById('next');
let currentIndex       = 0;

// Ensure slides line up in a row
slidesContainer.style.display    = 'flex';
slidesContainer.style.transition = 'transform 0.5s ease';

// Make each slide exactly the width of the slider section
function sizeSlides() {
  const w = sliderSection.clientWidth + 'px';
  slides.forEach(s => {
    s.style.minWidth = w;
    s.style.maxWidth = w;
  });
}
window.addEventListener('resize', sizeSlides);
sizeSlides();

// Shift to the slide at `currentIndex`
function showSlide(idx) {
  const offset = sliderSection.clientWidth * idx;
  slidesContainer.style.transform = `translateX(-${offset}px)`;
}

// Next / previous handlers
prevBtn.onclick = () => {
  currentIndex = (currentIndex - 1 + slides.length) % slides.length;
  showSlide(currentIndex);
};
nextBtn.onclick = () => {
  currentIndex = (currentIndex + 1) % slides.length;
  showSlide(currentIndex);
};

// Auto-advance every 30s
setInterval(() => {
  currentIndex = (currentIndex + 1) % slides.length;
  showSlide(currentIndex);
}, 30000);

// Start on slide 0
showSlide(0);
