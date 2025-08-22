// Minimal, clickable slider with dots & autoplay
document.addEventListener('DOMContentLoaded',()=>{
  const slider = document.querySelector('.slider');
  if(!slider) return;

  const track = slider.querySelector('.slides');
  const slides = Array.from(track.children);
  const prev = slider.querySelector('.prev');
  const next = slider.querySelector('.next');
  const dots = slider.querySelector('.dots');
  let i = 0;

  function go(n){
    i = (n+slides.length)%slides.length;
    track.style.transform = `translateX(${-i*100}%)`;
    dots.querySelectorAll('.dot').forEach((d,idx)=>d.classList.toggle('active', idx===i));
  }

  // dots
  dots.innerHTML = slides.map(() => '<span class="dot"></span>').join('');
  dots.querySelectorAll('.dot').forEach((d,idx)=>d.addEventListener('click',()=>go(idx)));
  prev.addEventListener('click',()=>go(i-1));
  next.addEventListener('click',()=>go(i+1));

  // autoplay
  const delay = parseInt(slider.getAttribute('data-autoplay')||'0',10);
  let t=null;
  function start(){ if(delay>0){ t=setInterval(()=>go(i+1), delay); } }
  function stop(){ if(t){ clearInterval(t); t=null; } }
  slider.addEventListener('mouseenter', stop);
  slider.addEventListener('mouseleave', start);

  // ensure slides are <a> so any click navigates
  slides.forEach(s => { s.style.pointerEvents='auto'; s.querySelectorAll('*').forEach(el=>el.style.pointerEvents='auto'); });

  go(0); start();
});
