// slider.js

const slidesContainer = document.getElementById('slides');
const slides = Array.from(document.querySelectorAll('.slide'));
const prevButton = document.getElementById('prev');
const nextButton = document.getElementById('next');
let currentIndex = 0;

function update() {
  slidesContainer.style.transform = `translateX(-${currentIndex * 100}%)`;
}

function showNext() {
  currentIndex = (currentIndex + 1) % slides.length;
  update();
}

function showPrev() {
  currentIndex = (currentIndex - 1 + slides.length) % slides.length;
  update();
}

nextButton.addEventListener('click', showNext);
prevButton.addEventListener('click', showPrev);

// Auto-advance every 30 seconds
setInterval(showNext, 30000);

// Initial setup
slidesContainer.style.display = 'flex';
slidesContainer.style.overflow = 'hidden';
slides.forEach(s => s.style.flex = '0 0 100%');
update();
