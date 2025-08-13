(function(){
  const slider = document.querySelector('.slider');
  if(!slider) return;

  const track = slider.querySelector('.slides');
  const btnPrev = slider.querySelector('.prev');
  const btnNext = slider.querySelector('.next');
  const dotsWrap = slider.querySelector('.dots');
  const autoplay = Number(slider.getAttribute('data-autoplay') || 7000);

  const slides = () => Array.from(slider.querySelectorAll('.slide'));
  let index = 0, timer = null, x0 = null, locked = false;

  function buildDots(){
    dotsWrap.innerHTML = '';
    slides().forEach((_,i)=>{
      const d = document.createElement('span');
      d.className = 'dot' + (i===index?' active':'');
      d.addEventListener('click', ()=>go(i));
      dotsWrap.appendChild(d);
    });
  }
  function go(i){
    const total = slides().length;
    index = (i+total)%total;
    track.style.transform = `translate3d(${-index*100}%,0,0)`;
    buildDots(); restart();
  }
  function next(){ go(index+1) }
  function prev(){ go(index-1) }
  function start(){ if(autoplay>0) timer = setInterval(next, autoplay) }
  function stop(){ if(timer){ clearInterval(timer); timer=null } }
  function restart(){ stop(); start(); }

  // Pointer drag
  track.addEventListener('pointerdown', (e)=>{ x0=e.clientX; locked=true; stop(); track.setPointerCapture(e.pointerId); });
  track.addEventListener('pointermove', (e)=>{ if(!locked) return; const dx=e.clientX-x0; track.style.transform=`translate3d(calc(${-index*100}% + ${dx}px),0,0)`; });
  track.addEventListener('pointerup', (e)=>{
    if(!locked) return; const dx=e.clientX-x0; locked=false; x0=null;
    const w = slider.clientWidth; if(Math.abs(dx)>w*0.18) (dx<0?next():prev()); else go(index); start();
  });

  btnPrev.addEventListener('click', prev);
  btnNext.addEventListener('click', next);
  window.addEventListener('resize', ()=>go(index));

  go(0); start();
})();
