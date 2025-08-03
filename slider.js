const slides = () => [...document.querySelectorAll('.slide')];
let idx = 0;

function show(n) {
  slides().forEach((s, i) => s.style.opacity = i === n ? 1 : 0);
}
function next() { idx = (idx + 1) % slides().length; show(idx); }
function prev() { idx = (idx || slides().length) - 1; show(idx); }

document.getElementById('next').onclick = next;
document.getElementById('prev').onclick = prev;

show(0);
setInterval(next, 30000); // 30-second auto-advance
