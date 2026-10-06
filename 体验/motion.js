// 动效只读取交互状态；不保存聊天内容，不影响模型推理。
const motionButton=document.getElementById('motion-toggle');
const reducedMotion=window.matchMedia('(prefers-reduced-motion: reduce)');
let motionEnabled=localStorage.getItem('xinyu-motion')!=='off'&&!reducedMotion.matches;
let animationFrame=0,lastFrame=0,sceneWidth=0,sceneHeight=0;
const canvas=document.getElementById('petals'),paint=canvas.getContext('2d');
const petals=Array.from({length:48},()=>({x:Math.random(),y:Math.random(),size:6+Math.random()*8,speed:.018+Math.random()*.025,phase:Math.random()*Math.PI*2}));
function resizeScene(){sceneWidth=window.innerWidth;sceneHeight=window.innerHeight;const scale=Math.min(window.devicePixelRatio||1,1.5);canvas.width=sceneWidth*scale;canvas.height=sceneHeight*scale;paint.setTransform(scale,0,0,scale,0,0);}
function animateScene(now){animationFrame=0;if(!motionEnabled||document.hidden)return;if(now-lastFrame>32){const delta=Math.min((now-lastFrame)/1000,.08);lastFrame=now;paint.clearRect(0,0,sceneWidth,sceneHeight);for(const petal of petals){petal.y+=petal.speed*delta;if(petal.y>1.04){petal.y=-.04;petal.x=Math.random();}const x=petal.x*sceneWidth+Math.sin(now*.0005+petal.phase)*25,y=petal.y*sceneHeight;paint.save();paint.translate(x,y);paint.rotate(now*.0003+petal.phase);paint.globalAlpha=petal.x>.38?.4:.9;paint.fillStyle='#ffb9d6';paint.strokeStyle='#fff3fa';paint.lineWidth=.8;paint.beginPath();paint.ellipse(0,0,petal.size,petal.size*.5,0,0,Math.PI*2);paint.fill();paint.stroke();paint.restore();}}animationFrame=requestAnimationFrame(animateScene);}
function syncMotion(){document.body.classList.toggle('motion-off',!motionEnabled);motionButton.setAttribute('aria-pressed',String(motionEnabled));motionButton.textContent=motionEnabled?'✧ 动效开启':'✧ 动效关闭';cancelAnimationFrame(animationFrame);animationFrame=0;paint.clearRect(0,0,sceneWidth,sceneHeight);if(motionEnabled&&!document.hidden){lastFrame=performance.now();animationFrame=requestAnimationFrame(animateScene);}}
motionButton.onclick=()=>{motionEnabled=!motionEnabled;localStorage.setItem('xinyu-motion',motionEnabled?'on':'off');syncMotion();};
document.addEventListener('visibilitychange',syncMotion);
reducedMotion.addEventListener('change',event=>{if(event.matches){motionEnabled=false;syncMotion();}});
window.addEventListener('resize',resizeScene);resizeScene();syncMotion();
