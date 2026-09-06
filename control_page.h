#pragma once
// Served from flash via server.send_P — keep this file as one PROGMEM page.

static const char CONTROL_PAGE[] PROGMEM = R"=====(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>AI Mini Bot — Face Deck</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@500;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0a0c0b;
  --panel:#121614;
  --line:#243028;
  --ink:#e8f0e6;
  --muted:#7f9186;
  --phosphor:#9dff7a;
  --warm:#ffc857;
  --danger:#ff6b4a;
  --glow:0 0 28px rgba(157,255,122,.22);
}
*{box-sizing:border-box}
html,body{margin:0;min-height:100%;background:var(--bg);color:var(--ink);font-family:"IBM Plex Mono",monospace}
body{
  background:
    radial-gradient(900px 500px at 10% -10%,rgba(157,255,122,.08),transparent 55%),
    radial-gradient(700px 420px at 100% 0%,rgba(255,200,87,.06),transparent 50%),
    linear-gradient(180deg,#0d100e 0%,#080a09 100%);
}
body::before{
  content:"";position:fixed;inset:0;pointer-events:none;opacity:.07;z-index:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}
.wrap{position:relative;z-index:1;max-width:980px;margin:0 auto;padding:28px 18px 64px}
.top{display:flex;flex-wrap:wrap;align-items:end;justify-content:space-between;gap:16px;margin-bottom:28px}
h1{font-family:Syne,sans-serif;font-weight:800;font-size:clamp(2rem,5vw,3.1rem);letter-spacing:-.04em;margin:0;line-height:.95}
h1 span{color:var(--phosphor);text-shadow:var(--glow)}
.sub{color:var(--muted);font-size:12px;margin-top:8px;max-width:34ch;line-height:1.5}
.pill{
  display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border:1px solid var(--line);
  background:rgba(18,22,20,.85);border-radius:999px;font-size:12px;color:var(--muted)
}
.pill b{color:var(--phosphor);font-weight:500}
.stage{
  display:grid;grid-template-columns:1.1fr .9fr;gap:18px;margin-bottom:28px
}
@media(max-width:780px){.stage{grid-template-columns:1fr}}
.panel{
  background:linear-gradient(165deg,rgba(24,30,26,.95),rgba(14,17,15,.98));
  border:1px solid var(--line);border-radius:22px;padding:18px;box-shadow:inset 0 1px 0 rgba(255,255,255,.03)
}
.panel h2{font-family:Syne,sans-serif;font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 14px;font-weight:700}
.preview-shell{
  aspect-ratio:16/10;border-radius:18px;background:#050705;border:1px solid #1c281f;
  display:grid;place-items:center;position:relative;overflow:hidden;box-shadow:var(--glow)
}
.preview-shell::after{
  content:"";position:absolute;inset:0;background:linear-gradient(180deg,transparent 55%,rgba(0,0,0,.35));
  pointer-events:none
}
.oled{
  width:min(280px,78%);aspect-ratio:2/1;background:#020302;border-radius:10px;
  border:2px solid #1a221c;position:relative;overflow:hidden
}
.oled .scan{position:absolute;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(157,255,122,.035) 3px);pointer-events:none}
.face{position:absolute;inset:0;display:grid;place-items:center}
.eyes{position:relative;width:78%;height:58%;display:flex;justify-content:space-between;align-items:center}
.eye{
  width:38%;height:86%;background:var(--phosphor);border-radius:18px;
  box-shadow:0 0 18px rgba(157,255,122,.35);position:relative;overflow:hidden
}
.live-label{position:absolute;left:14px;top:12px;font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.live-name{font-family:Syne,sans-serif;font-size:28px;font-weight:700;margin:14px 0 4px;letter-spacing:-.03em}
.live-hint{color:var(--muted);font-size:12px;line-height:1.45}
.grid{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin-bottom:28px
}
.card{
  appearance:none;border:1px solid var(--line);background:var(--panel);border-radius:18px;padding:12px;
  color:var(--ink);cursor:pointer;text-align:left;transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease;
  font:inherit
}
.card:hover{transform:translateY(-2px);border-color:#3a5242;box-shadow:0 10px 30px rgba(0,0,0,.25)}
.card.active{border-color:var(--phosphor);box-shadow:var(--glow)}
.card .mini{
  height:72px;border-radius:12px;background:#050705;border:1px solid #1a221c;
  display:grid;place-items:center;margin-bottom:10px;overflow:hidden;position:relative
}
.card .mini .eyes{width:72%;height:48%}
.card .mini .eye{border-radius:10px;box-shadow:none}
.card strong{display:block;font-family:Syne,sans-serif;font-size:14px;font-weight:700;letter-spacing:-.02em;text-transform:capitalize}
.card em{display:block;margin-top:3px;font-style:normal;color:var(--muted);font-size:10px;letter-spacing:.04em}
.head-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
.head-grid button,.action{
  appearance:none;border:1px solid var(--line);background:#171c19;color:var(--ink);
  border-radius:12px;padding:12px 8px;font:inherit;font-size:12px;cursor:pointer;letter-spacing:.04em;text-transform:uppercase
}
.head-grid button:hover,.action:hover{border-color:var(--warm);color:var(--warm)}
label.row{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin:10px 0 6px}
input[type=range]{width:100%;accent-color:var(--phosphor)}
.toast{min-height:1.2em;margin-top:12px;color:var(--phosphor);font-size:12px}

/* emotion animations on .face[data-mood] */
.face[data-mood=neutral] .eye{animation:blink 3.6s infinite}
.face[data-mood=listening] .eye{animation:pulseH 1.1s ease-in-out infinite}
.face[data-mood=talking] .eyes{animation:bob 0.35s ease-in-out infinite}
.face[data-mood=happy] .eye{
  height:52%;border-radius:50% 50% 8px 8px;align-self:end;margin-top:18%;
  clip-path:ellipse(50% 55% at 50% 35%)
}
.face[data-mood=sad] .eye{height:62%;transform:translateY(10%);border-radius:14px}
.face[data-mood=sad] .eye::after{
  content:"";position:absolute;left:0;top:0;width:100%;height:45%;background:#020302;
  clip-path:polygon(0 0,100% 0,100% 100%,0 40%)
}
.face[data-mood=angry] .eye::after{
  content:"";position:absolute;right:0;top:0;width:70%;height:48%;background:#020302;
  clip-path:polygon(100% 0,0 0,100% 100%)
}
.face[data-mood=angry] .eye:first-child::after{left:0;right:auto;clip-path:polygon(0 0,100% 0,0 100%)}
.face[data-mood=surprised] .eye{border-radius:50%;height:92%}
.face[data-mood=surprised] .eye::after{
  content:"";position:absolute;inset:28%;background:#020302;border-radius:50%
}
.face[data-mood=thinking] .eyes{transform:translate(10%,-8%);height:48%}
.face[data-mood=sleep] .eye{height:10%;border-radius:6px}
.face[data-mood=sleep]::after{
  content:"z z";position:absolute;right:10%;top:12%;color:var(--phosphor);font-size:14px;opacity:.8;
  animation:floatZ 1.8s ease-in-out infinite
}
.face[data-mood=searching] .eyes{animation:dart 1.1s ease-in-out infinite}
.face[data-mood=loading] .eyes{display:none}
.face[data-mood=loading]::before{
  content:"";width:42px;height:42px;border-radius:50%;
  border:3px solid rgba(157,255,122,.15);border-top-color:var(--phosphor);
  animation:spin .8s linear infinite
}
.face[data-mood=scanning] .eyes{height:36%}
.face[data-mood=scanning]::after{
  content:"";position:absolute;top:0;bottom:0;width:2px;background:var(--phosphor);
  box-shadow:0 0 10px var(--phosphor);animation:scan 1.8s ease-in-out infinite
}
.face[data-mood=wifi] .eyes{display:none}
.face[data-mood=wifi]::before{
  content:"";width:10px;height:10px;border-radius:50%;background:var(--phosphor);
  box-shadow:0 -14px 0 -3px var(--phosphor),0 -26px 0 -5px var(--phosphor),0 -36px 0 -7px rgba(157,255,122,.45);
  animation:wifi 1.2s steps(3) infinite;margin-top:28px
}
.face[data-mood=memory] .eyes{transform:translate(-8%,-10%);height:48%}
.face[data-mood=memory]::after{
  content:"";position:absolute;right:14%;top:18%;width:6px;height:6px;border-radius:50%;background:var(--phosphor);
  box-shadow:10px -10px 0 0 var(--phosphor),20px -20px 0 -1px var(--phosphor);animation:dots 1.2s infinite
}
.face[data-mood=saving] .eyes{height:42%;transform:translateY(-12%)}
.face[data-mood=saving]::after{
  content:"";position:absolute;left:18%;right:18%;bottom:14%;height:8px;border:1px solid var(--phosphor);border-radius:4px;
  background:linear-gradient(90deg,var(--phosphor) 40%,transparent 41%);background-size:200% 100%;
  animation:bar 1.4s linear infinite
}

@keyframes blink{0%,40%,44%,100%{transform:scaleY(1)}42%{transform:scaleY(.08)}}
@keyframes pulseH{0%,100%{transform:scaleY(1)}50%{transform:scaleY(1.12)}}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(3px)}}
@keyframes dart{0%,100%{transform:translateX(-12px)}50%{transform:translateX(12px)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes scan{0%,100%{left:18%}50%{left:80%}}
@keyframes wifi{0%{opacity:.35}100%{opacity:1}}
@keyframes dots{0%,100%{opacity:.35}50%{opacity:1}}
@keyframes bar{to{background-position:-200% 0}}
@keyframes floatZ{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
.card,.panel,.top>*{animation:rise .55s ease both}
.card:nth-child(1){animation-delay:.02s}.card:nth-child(2){animation-delay:.04s}
.card:nth-child(3){animation-delay:.06s}.card:nth-child(4){animation-delay:.08s}
.card:nth-child(5){animation-delay:.1s}.card:nth-child(6){animation-delay:.12s}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <h1>face <span>deck</span></h1>
      <p class="sub">Tap a mood to push it to the OLED. Live preview mirrors the robot's expression set.</p>
    </div>
    <div class="pill">status <b id="status">ready</b></div>
  </div>

  <div class="stage">
    <section class="panel">
      <h2>Live preview</h2>
      <div class="preview-shell">
        <div class="live-label">oled · 128×128</div>
        <div class="oled">
          <div class="face" id="live" data-mood="neutral">
            <div class="eyes"><div class="eye"></div><div class="eye"></div></div>
          </div>
          <div class="scan"></div>
        </div>
      </div>
      <div class="live-name" id="liveName">Neutral</div>
      <div class="live-hint" id="liveHint">Calm resting face with a soft blink.</div>
    </section>

    <section class="panel">
      <h2>Head · pan / tilt</h2>
      <div class="head-grid">
        <button type="button" onclick="go(45,90)">left</button>
        <button type="button" onclick="go(90,90)">center</button>
        <button type="button" onclick="go(135,90)">right</button>
        <button type="button" onclick="go(90,60)">up</button>
        <button type="button" onclick="go(90,120)">down</button>
        <button type="button" class="action" onclick="snap()">camera</button>
      </div>
      <label class="row">pan <span id="pv">90</span></label>
      <input id="pan" type="range" min="15" max="165" value="90">
      <label class="row">tilt <span id="tv">90</span></label>
      <input id="tilt" type="range" min="30" max="150" value="90">

      <div class="head-grid" style="margin-top:14px">
        <button type="button" onclick="beep(660)">beep lo</button>
        <button type="button" onclick="beep(1047)">beep hi</button>
        <button type="button" onclick="hush()">stop</button>
      </div>
      <label class="row">volume <span id="vv">6</span></label>
      <input id="vol" type="range" min="0" max="10" value="6">

      <div class="toast" id="toast">waiting for command…</div>
      <div id="camWrap" style="display:none;width:100%;aspect-ratio:1/1;margin-top:12px;border-radius:12px;border:1px solid var(--line);overflow:hidden;background:#000;align-items:center;justify-content:center">
        <img id="cam" alt="" style="max-width:100%;max-height:100%;display:block;transform:rotate(90deg)">
      </div>
    </section>
  </div>

  <div class="grid" id="faces"></div>
</div>
<script>
const FACES=[
  {id:'neutral',hint:'Calm resting face with a soft blink.'},
  {id:'happy',hint:'Squinted smile eyes — good news energy.'},
  {id:'sad',hint:'Soft downturn, a little wilted.'},
  {id:'angry',hint:'Hard brows, don\u2019t test it.'},
  {id:'surprised',hint:'Wide pupils, caught mid-gasp.'},
  {id:'thinking',hint:'Eyes drift up while it works it out.'},
  {id:'listening',hint:'Gentle pulse — it\u2019s paying attention.'},
  {id:'talking',hint:'A tiny bob while it speaks.'},
  {id:'sleep',hint:'Almost shut, with floating z\u2019s.'},
  {id:'searching',hint:'Eyes dart while it hunts the web.'},
  {id:'loading',hint:'Spinner ring for opening apps.'},
  {id:'scanning',hint:'Sweep bar across a focused stare.'},
  {id:'wifi',hint:'Signal arcs climbing from a node.'},
  {id:'memory',hint:'Glance up with rising thought dots.'},
  {id:'saving',hint:'Progress bar while a note lands.'}
];
const live=document.getElementById('live');
const liveName=document.getElementById('liveName');
const liveHint=document.getElementById('liveHint');
const statusEl=document.getElementById('status');
const toast=document.getElementById('toast');
const faces=document.getElementById('faces');
const pan=document.getElementById('pan'), tilt=document.getElementById('tilt');
const pv=document.getElementById('pv'), tv=document.getElementById('tv');
const vol=document.getElementById('vol'), vv=document.getElementById('vv');
let current='neutral';

function eyeHTML(){return '<div class="eyes"><div class="eye"></div><div class="eye"></div></div>'}
function setPreview(id){
  const f=FACES.find(x=>x.id===id)||FACES[0];
  current=id;
  live.dataset.mood=id;
  live.innerHTML=eyeHTML();
  liveName.textContent=id;
  liveHint.textContent=f.hint;
  document.querySelectorAll('.card').forEach(c=>c.classList.toggle('active',c.dataset.id===id));
}
function setFace(id){
  setPreview(id);
  statusEl.textContent='sending…';
  fetch('/set?e='+encodeURIComponent(id))
    .then(r=>r.text())
    .then(t=>{statusEl.textContent=t;toast.textContent=t})
    .catch(e=>{statusEl.textContent='error';toast.textContent=String(e)});
}
function go(p,t){
  pan.value=p;tilt.value=t;pv.textContent=p;tv.textContent=t;
  fetch('/look?pan='+p+'&tilt='+t).then(r=>r.text()).then(x=>toast.textContent=x).catch(e=>toast.textContent=String(e));
}
function snap(){
  const wrap=document.getElementById('camWrap');
  const img=document.getElementById('cam');
  wrap.style.display='flex';
  img.src='/capture?'+Date.now();
  toast.textContent='camera grab…';
}
function beep(f){
  fetch('/beep?f='+f+'&ms=180').then(r=>r.text()).then(x=>toast.textContent='beep '+f+'Hz').catch(e=>toast.textContent=String(e));
}
function hush(){
  fetch('/stop').then(r=>r.text()).then(x=>toast.textContent='speech stopped').catch(e=>toast.textContent=String(e));
}
let timer=null;
function slide(){
  pv.textContent=pan.value;tv.textContent=tilt.value;
  clearTimeout(timer);
  timer=setTimeout(()=>go(+pan.value,+tilt.value),120);
}
pan.addEventListener('input',slide);tilt.addEventListener('input',slide);

let vtimer=null;
vol.addEventListener('input',()=>{
  vv.textContent=vol.value;
  clearTimeout(vtimer);
  vtimer=setTimeout(()=>fetch('/volume?v='+vol.value).then(r=>r.text()).then(x=>toast.textContent=x),150);
});

FACES.forEach((f,i)=>{
  const b=document.createElement('button');
  b.type='button';b.className='card'+(i===0?' active':'');b.dataset.id=f.id;
  b.innerHTML='<div class="mini"><div class="face" data-mood="'+f.id+'">'+eyeHTML()+'</div></div><strong>'+f.id+'</strong><em>'+f.hint+'</em>';
  b.onclick=()=>setFace(f.id);
  faces.appendChild(b);
});
setPreview('neutral');

setInterval(()=>{
  fetch('/status').then(r=>r.json()).then(s=>{
    statusEl.textContent=
      (s.camera?'cam ':'') + (s.audio?'spk ':'') + (s.mic?'mic ':'') +
      (s.servos?'srv ':'') + s.rssi+'dBm';
  }).catch(()=>{statusEl.textContent='offline'});
},5000);
</script>
</body>
</html>
)=====";
