/**
 * AlgoChat — Enhanced Particle Engine
 * 复古未来主义粒子特效：多层视差 + 色相循环 + 辉光 + 流星 + 鼠标交互
 * 酷炫但不喧宾夺主
 */

class ParticleEngine {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.particles = [];
    this.shootingStars = [];
    this.mouse = { x: -1000, y: -1000, active: false };
    this.running = true;
    this.time = 0;
    this.hueOffset = 0;

    // Theme colors in HSL (approximated from RGB)
    this.palette = [
      { h: 155, s: 14, l: 55 },  // teal (125,158,141)
      { h: 25,  s: 58, l: 35 },  // orange (142,78,38)
      { h: 8,   s: 67, l: 36 },  // red (153,45,30)
      { h: 165, s: 10, l: 70 },  // teal-light
      { h: 35,  s: 40, l: 60 },  // warm neutral
    ];

    this.resize();
    this.initParticles();
    this.bindEvents();
    this.animate();
  }

  resize() {
    this.width = this.canvas.width = window.innerWidth;
    this.height = this.canvas.height = window.innerHeight;
  }

  bindEvents() {
    window.addEventListener('resize', () => {
      this.resize();
      this.initParticles();
    });

    window.addEventListener('mousemove', (e) => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
      this.mouse.active = true;
    });

    window.addEventListener('mouseleave', () => {
      this.mouse.active = false;
    });
  }

  initParticles() {
    this.particles = [];
    const area = this.width * this.height;
    const density = Math.min(Math.max(area / 12000, 50), 120);
    const count = Math.floor(density);

    for (let i = 0; i < count; i++) {
      const layer = Math.random() < 0.25 ? 0 : (Math.random() < 0.55 ? 1 : 2);
      this.particles.push(this.createParticle(layer));
    }
  }

  createParticle(layer, x, y) {
    const colorIdx = Math.floor(Math.random() * this.palette.length);
    const baseColor = this.palette[colorIdx];

    // Shapes: circle, diamond, hexagon, ring, triangle
    const shapes = ['circle', 'circle', 'circle', 'diamond', 'hexagon', 'ring', 'triangle'];
    const shape = shapes[Math.floor(Math.random() * shapes.length)];

    // Layer properties: 0=far, 1=mid, 2=near
    const layerConfig = [
      { sizeRange: [1.5, 3], speedRange: [0.08, 0.2], opacity: 0.12, blur: 0 },
      { sizeRange: [2, 4.5], speedRange: [0.15, 0.35], opacity: 0.18, blur: 2 },
      { sizeRange: [3, 6], speedRange: [0.25, 0.5], opacity: 0.22, blur: 4 },
    ][layer];

    const size = layerConfig.sizeRange[0] + Math.random() * (layerConfig.sizeRange[1] - layerConfig.sizeRange[0]);
    const speed = layerConfig.speedRange[0] + Math.random() * (layerConfig.speedRange[1] - layerConfig.speedRange[0]);
    const angle = Math.random() * Math.PI * 2;

    return {
      x: x !== undefined ? x : Math.random() * this.width,
      y: y !== undefined ? y : Math.random() * this.height,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      size,
      layer,
      shape,
      baseColor,
      opacity: layerConfig.opacity * (0.6 + Math.random() * 0.4),
      glowBlur: layerConfig.blur,
      rotation: Math.random() * Math.PI * 2,
      rotationSpeed: (Math.random() - 0.5) * 0.01,
      pulsePhase: Math.random() * Math.PI * 2,
      pulseSpeed: 0.005 + Math.random() * 0.01,
    };
  }

  spawnShootingStar() {
    if (this.shootingStars.length >= 2) return;
    if (Math.random() > 0.003) return; // ~every 8-15 seconds at 60fps

    const startX = Math.random() * this.width * 0.8;
    const startY = Math.random() * this.height * 0.3;
    const angle = Math.PI / 6 + Math.random() * Math.PI / 6;
    const speed = 4 + Math.random() * 3;

    this.shootingStars.push({
      x: startX,
      y: startY,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      life: 1,
      decay: 0.015 + Math.random() * 0.01,
      trail: [],
      maxTrail: 15 + Math.floor(Math.random() * 10),
    });
  }

  updateParticle(p) {
    // Slow color cycling
    p.pulsePhase += p.pulseSpeed;
    const pulse = 0.85 + 0.15 * Math.sin(p.pulsePhase);

    // Mouse interaction: gentle attract then repel at very close range
    if (this.mouse.active) {
      const dx = this.mouse.x - p.x;
      const dy = this.mouse.y - p.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const influenceRadius = 120 + p.layer * 30;

      if (dist < influenceRadius && dist > 1) {
        const force = (1 - dist / influenceRadius) * 0.15;
        if (dist > 40) {
          // Attract gently
          p.vx += (dx / dist) * force * 0.3;
          p.vy += (dy / dist) * force * 0.3;
        } else {
          // Repel softly at very close range
          p.vx -= (dx / dist) * force * 0.5;
          p.vy -= (dy / dist) * force * 0.5;
        }
      }
    }

    // Damping
    p.vx *= 0.995;
    p.vy *= 0.995;

    // Speed limit
    const maxSpeed = 0.6 + p.layer * 0.2;
    const currentSpeed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
    if (currentSpeed > maxSpeed) {
      p.vx = (p.vx / currentSpeed) * maxSpeed;
      p.vy = (p.vy / currentSpeed) * maxSpeed;
    }

    p.x += p.vx;
    p.y += p.vy;
    p.rotation += p.rotationSpeed;

    // Wrap around edges with margin
    const margin = 20;
    if (p.x < -margin) p.x = this.width + margin;
    if (p.x > this.width + margin) p.x = -margin;
    if (p.y < -margin) p.y = this.height + margin;
    if (p.y > this.height + margin) p.y = -margin;

    p.currentOpacity = p.opacity * pulse;
    p.currentSize = p.size * (0.9 + 0.1 * pulse);
  }

  drawParticle(p) {
    const ctx = this.ctx;
    const size = p.currentSize;

    // Hue cycling (very subtle)
    const hueShift = Math.sin(this.hueOffset + p.pulsePhase * 0.3) * 10;
    const h = p.baseColor.h + hueShift;
    const s = p.baseColor.s;
    const l = p.baseColor.l;

    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rotation);
    ctx.globalAlpha = p.currentOpacity;

    // Glow effect
    if (p.glowBlur > 0) {
      ctx.shadowBlur = p.glowBlur;
      ctx.shadowColor = `hsla(${h}, ${s}%, ${l}%, 0.3)`;
    }

    ctx.fillStyle = `hsla(${h}, ${s}%, ${l}%, ${p.currentOpacity})`;
    ctx.strokeStyle = `hsla(${h}, ${s}%, ${l}%, ${p.currentOpacity * 0.6})`;
    ctx.lineWidth = 0.5;

    switch (p.shape) {
      case 'circle':
        ctx.beginPath();
        ctx.arc(0, 0, size, 0, Math.PI * 2);
        ctx.fill();
        break;

      case 'diamond':
        ctx.beginPath();
        ctx.moveTo(0, -size);
        ctx.lineTo(size * 0.7, 0);
        ctx.lineTo(0, size);
        ctx.lineTo(-size * 0.7, 0);
        ctx.closePath();
        ctx.fill();
        break;

      case 'hexagon':
        ctx.beginPath();
        for (let i = 0; i < 6; i++) {
          const angle = (Math.PI / 3) * i - Math.PI / 6;
          const px = Math.cos(angle) * size;
          const py = Math.sin(angle) * size;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fill();
        break;

      case 'ring':
        ctx.beginPath();
        ctx.arc(0, 0, size, 0, Math.PI * 2);
        ctx.stroke();
        break;

      case 'triangle':
        ctx.beginPath();
        ctx.moveTo(0, -size);
        ctx.lineTo(size * 0.87, size * 0.5);
        ctx.lineTo(-size * 0.87, size * 0.5);
        ctx.closePath();
        ctx.fill();
        break;
    }

    ctx.restore();
  }

  drawConnections() {
    const ctx = this.ctx;
    const maxDist = 160;

    for (let i = 0; i < this.particles.length; i++) {
      for (let j = i + 1; j < this.particles.length; j++) {
        const a = this.particles[i];
        const b = this.particles[j];

        if (Math.abs(a.layer - b.layer) > 1) continue;

        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < maxDist) {
          const t = 1 - dist / maxDist;
          const alpha = t * 0.12;
          const hue = (a.baseColor.h + b.baseColor.h) / 2 + Math.sin(this.hueOffset + i * 0.1) * 8;

          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `hsla(${hue}, 18%, 55%, ${alpha})`;
          ctx.lineWidth = t * 1.2;
          ctx.stroke();
        }
      }
    }

    // Mouse-range connections
    if (this.mouse.active) {
      const mouseRadius = 200;
      const nearParticles = this.particles.filter(p => {
        const dx = p.x - this.mouse.x;
        const dy = p.y - this.mouse.y;
        return dx * dx + dy * dy < mouseRadius * mouseRadius;
      });

      for (let i = 0; i < nearParticles.length; i++) {
        const a = nearParticles[i];
        const dxm = a.x - this.mouse.x;
        const dym = a.y - this.mouse.y;
        const distm = Math.sqrt(dxm * dxm + dym * dym);
        const alphaM = (1 - distm / mouseRadius) * 0.2;

        ctx.beginPath();
        ctx.moveTo(this.mouse.x, this.mouse.y);
        ctx.lineTo(a.x, a.y);
        ctx.strokeStyle = `hsla(35, 50%, 60%, ${alphaM})`;
        ctx.lineWidth = 0.8;
        ctx.stroke();

        for (let j = i + 1; j < nearParticles.length; j++) {
          const b = nearParticles[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < maxDist * 1.3) {
            const alpha = (1 - dist / (maxDist * 1.3)) * 0.18;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = `hsla(35, 35%, 60%, ${alpha})`;
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }
        }
      }
    }
  }

  updateShootingStar(s) {
    s.trail.unshift({ x: s.x, y: s.y });
    if (s.trail.length > s.maxTrail) s.trail.pop();

    s.x += s.vx;
    s.y += s.vy;
    s.life -= s.decay;
  }

  drawShootingStar(s) {
    const ctx = this.ctx;
    if (s.trail.length < 2) return;

    for (let i = 1; i < s.trail.length; i++) {
      const alpha = s.life * (1 - i / s.trail.length) * 0.35;
      const width = (1 - i / s.trail.length) * 1.5;

      ctx.beginPath();
      ctx.moveTo(s.trail[i - 1].x, s.trail[i - 1].y);
      ctx.lineTo(s.trail[i].x, s.trail[i].y);
      ctx.strokeStyle = `hsla(35, 40%, 60%, ${alpha})`;
      ctx.lineWidth = width;
      ctx.stroke();
    }

    // Head glow
    ctx.save();
    ctx.globalAlpha = s.life * 0.5;
    ctx.shadowBlur = 8;
    ctx.shadowColor = 'hsla(35, 50%, 65%, 0.4)';
    ctx.beginPath();
    ctx.arc(s.x, s.y, 1.5, 0, Math.PI * 2);
    ctx.fillStyle = 'hsla(35, 50%, 75%, 0.6)';
    ctx.fill();
    ctx.restore();
  }

  animate() {
    if (!this.running) {
      requestAnimationFrame(() => this.animate());
      return;
    }

    this.time++;
    this.hueOffset += 0.001;

    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    // Update and draw particles (far layer first)
    const sorted = [...this.particles].sort((a, b) => a.layer - b.layer);

    for (const p of sorted) {
      this.updateParticle(p);
    }

    // Draw connections
    this.drawConnections();

    // Draw particles
    for (const p of sorted) {
      this.drawParticle(p);
    }

    // Shooting stars
    this.spawnShootingStar();
    this.shootingStars = this.shootingStars.filter(s => s.life > 0);
    for (const s of this.shootingStars) {
      this.updateShootingStar(s);
      this.drawShootingStar(s);
    }

    requestAnimationFrame(() => this.animate());
  }

  stop() {
    this.running = false;
  }

  start() {
    this.running = true;
  }

  destroy() {
    this.running = false;
    this.particles = [];
    this.shootingStars = [];
    this.ctx.clearRect(0, 0, this.width, this.height);
  }
}

// ══════════════════════════════════════════
// GEOMETRIC LINES INTRO (retro-futuristic, lightweight)
// ══════════════════════════════════════════
class GeoIntro {
  constructor(canvas, onComplete) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onComplete = onComplete;
    this.mouse = { x: -1000, y: -1000 };
    this.startTime = performance.now();
    this.duration = 3200;
    this.running = true;

    // Raise canvas above splash screen during intro
    this._origZIndex = canvas.style.zIndex || '';
    canvas.style.zIndex = '10000';
    canvas.style.pointerEvents = 'none';
    canvas.style.opacity = '1';

    // Make splash transparent so canvas shows through
    const splash = document.getElementById('splashScreen');
    if (splash) {
      splash.style.background = 'transparent';
      splash.classList.add('intro-active');
    }

    this.gridLines = [];   // precompute grid data
    this.resize();
    this.bindEvents();
    this.buildGrid();
    this.animate();
  }

  resize() {
    this.w = this.canvas.width = window.innerWidth;
    this.h = this.canvas.height = window.innerHeight;
  }

  bindEvents() {
    this._onResize = () => {
      this.resize();
      this.buildGrid();
    };
    this._onMouseMove = (e) => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
    };
    window.addEventListener('resize', this._onResize);
    window.addEventListener('mousemove', this._onMouseMove);
  }

  buildGrid() {
    // Build a sparse grid of nodes for a "circuit board" look
    this.gridLines = [];
    const step = Math.max(40, Math.min(this.w, this.h) / 28);
    const cols = Math.floor(this.w / step) + 1;
    const rows = Math.floor(this.h / step) + 1;

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const x = col * step;
        const y = row * step;
        if (col + 1 < cols) this.gridLines.push({ x1: x, y1: y, x2: x + step, y2: y });
        if (row + 1 < rows) this.gridLines.push({ x1: x, y1: y, x2: x, y2: y + step });
        if (col + 1 < cols && row + 1 < rows && Math.random() < 0.35) {
          this.gridLines.push({ x1: x, y1: y, x2: x + step, y2: y + step });
        }
      }
    }
  }

  animate() {
    if (!this.running) return;
    const elapsed = performance.now() - this.startTime;
    const progress = Math.min(elapsed / this.duration, 1);
    this.render(progress, elapsed);
    if (progress >= 1) this.complete();
    else requestAnimationFrame(() => this.animate());
  }

  render(progress, t) {
    const { ctx, w, h, mouse } = this;
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const bg = isDark ? '#0E0E0C' : '#F5F2ED';
    const teal = 'rgba(125,158,141,';
    const gold = 'rgba(196,155,96,';
    const warm = 'rgba(142,78,38,';

    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    // ─── Envelope curves ───
    const easeIn = Math.min(progress / 0.3, 1);       // 0–30%
    const phase2 = progress > 0.25 ? Math.min((progress - 0.25) / 0.35, 1) : 0; // 25–60%
    const phase3 = progress > 0.55 ? Math.min((progress - 0.55) / 0.3, 1) : 0;  // 55–85%
    const fadeOut = progress > 0.82 ? (progress - 0.82) / 0.18 : 0;              // 82–100%
    const overallAlpha = 1 - fadeOut;

    // Smooth in/out easing
    const ease = t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

    const cx = w / 2, cy = h / 2;
    const maxR = Math.sqrt(w * w + h * h) * 0.65;

    ctx.save();
    ctx.globalAlpha = overallAlpha;

    // ══════════════════════════════════════════
    // PHASE 1: Grid emergence (0–30%)
    // ══════════════════════════════════════════
    const gridAlpha = overallAlpha * (1 - ease(phase2)) * easeIn;
    if (gridAlpha > 0.005) {
      const rFade = easeIn * maxR;   // radius of visible grid
      ctx.lineWidth = 0.6;

      for (const ln of this.gridLines) {
        const lx = (ln.x1 + ln.x2) / 2;
        const ly = (ln.y1 + ln.y2) / 2;
        const dr = Math.hypot(lx - cx, ly - cy);
        if (dr > rFade) continue;

        const distAlpha = 1 - dr / rFade;
        const alpha = gridAlpha * distAlpha * 0.22;

        ctx.beginPath();
        ctx.moveTo(ln.x1, ln.y1);
        ctx.lineTo(ln.x2, ln.y2);
        ctx.strokeStyle = `${teal}${alpha.toFixed(3)})`;
        ctx.stroke();
      }

      // Mouse proximity: highlight local grid lines
      const mDist = Math.hypot(mouse.x - cx, mouse.y - cy);
      if (mDist < w && mouse.x > 0) {
        const mouseR = 160;
        for (const ln of this.gridLines) {
          const lx = (ln.x1 + ln.x2) / 2, ly = (ln.y1 + ln.y2) / 2;
          const dm = Math.hypot(lx - mouse.x, ly - mouse.y);
          if (dm < mouseR) {
            const ma = gridAlpha * (1 - dm / mouseR) * 0.3;
            ctx.beginPath();
            ctx.moveTo(ln.x1, ln.y1);
            ctx.lineTo(ln.x2, ln.y2);
            ctx.strokeStyle = `${gold}${ma.toFixed(3)})`;
            ctx.stroke();
          }
        }
      }
    }

    // ══════════════════════════════════════════
    // PHASE 2: Concentric polygons (25–60%) — "wave rings"
    // ══════════════════════════════════════════
    const ringAlpha = overallAlpha * ease(phase2) * (1 - phase3 * 0.7);
    if (ringAlpha > 0.005) {
      const ringCount = 8;
      const ringDelay = 0.08;
      const baseR = maxR * 0.5 * ease(phase2);

      for (let i = 0; i < ringCount; i++) {
        const riDelay = clamp((phase2 - i * ringDelay) / ringDelay, 0, 1);
        const riPhase = ease(riDelay);
        if (riPhase < 0.01) continue;

        const ri = 50 + i * (maxR / ringCount) * riPhase + Math.sin(t * 0.004 + i * 0.7) * 15 * riPhase;
        const ai = ringAlpha * riPhase * (1 - i / ringCount * 0.7);

        // Alternating: teal then gold then line-style
        const isTeal = i % 3 === 0;
        const isGold = i % 3 === 1;
        const color = isTeal ? teal : isGold ? gold : warm;
        const sides = i % 2 === 0 ? 6 : 8;

        // Animate rotation
        const rot = t * 0.0005 * (i % 2 === 0 ? 1 : -1) + i * 0.3;

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(rot);
        ctx.lineWidth = 0.7 + i * 0.04;
        ctx.beginPath();
        for (let v = 0; v <= sides; v++) {
          const a = (v / sides) * Math.PI * 2;
          const rx = Math.cos(a) * ri;
          const ry = Math.sin(a) * ri * 0.72; // slight flatten
          if (v === 0) ctx.moveTo(rx, ry);
          else ctx.lineTo(rx, ry);
        }
        ctx.closePath();
        ctx.strokeStyle = `${color}${ai.toFixed(3)})`;
        ctx.stroke();

        // Dotted overlay
        if (i % 3 === 2) {
          ctx.setLineDash([6, 12]);
          ctx.lineWidth = 0.4;
          ctx.strokeStyle = `${teal}${(ai * 0.5).toFixed(3)})`;
          ctx.stroke();
          ctx.setLineDash([]);
        }

        ctx.restore();
      }
    }

    // ══════════════════════════════════════════
    // PHASE 3: Radial convergence lines (55–85%)
    // ══════════════════════════════════════════
    const rayAlpha = overallAlpha * ease(phase3);
    if (rayAlpha > 0.005) {
      const rayCount = 36;
      const rayLen = maxR * 0.42 * ease(phase3);

      for (let i = 0; i < rayCount; i++) {
        const a = (i / rayCount) * Math.PI * 2 + t * 0.0003;
        const ox = cx + Math.cos(a) * rayLen * 0.08;
        const oy = cy + Math.sin(a) * rayLen * 0.08;
        const ex = cx + Math.cos(a) * rayLen;
        const ey = cy + Math.sin(a) * rayLen * 0.72;

        const ai = rayAlpha * (0.3 + 0.7 * Math.abs(Math.sin(i * 0.5)));

        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(ex, ey);
        ctx.lineWidth = 0.3 + (i % 4 === 0 ? 0.6 : 0);
        const rc = i % 4 === 0 ? gold : teal;
        ctx.strokeStyle = `${rc}${ai.toFixed(3)})`;
        ctx.stroke();
      }

      // Mouse attract: nearby rays bend toward mouse
      if (mouse.x > 0) {
        ctx.lineWidth = 0.5;
        ctx.strokeStyle = `${gold}${(rayAlpha * 0.25).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(mouse.x, mouse.y, 8 * ease(phase3), 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // ══════════════════════════════════════════
    // LOGO OVERLAY (0–88%) — fades in/out smoothly
    // ══════════════════════════════════════════
    if (progress < 0.88) {
      const logoIn = easeIn;
      const logoOut = progress > 0.72 ? (progress - 0.72) / 0.16 : 0;
      const logoAlpha = logoIn * (1 - ease(logoOut));
      if (logoAlpha > 0.02) {
        ctx.save();
        ctx.globalAlpha = logoAlpha;

        // Radial glow
        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 90);
        grad.addColorStop(0, 'rgba(125,158,141,0.18)');
        grad.addColorStop(1, 'rgba(125,158,141,0)');
        ctx.fillStyle = grad;
        ctx.fillRect(cx - 120, cy - 120, 240, 240);

        ctx.font = '600 54px "Playfair Display", serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = isDark ? '#FAFCF5' : '#1C1917';
        ctx.shadowBlur = 30;
        ctx.shadowColor = 'rgba(125,158,141,0.6)';
        ctx.fillText('◈', cx, cy - 30);

        ctx.font = '300 22px "Inter", sans-serif';
        ctx.shadowBlur = 15;
        ctx.fillText('AlgoChat', cx, cy + 28);
        ctx.restore();
      }
    }

    ctx.restore();
  }

  complete() {
    this.running = false;
    window.removeEventListener('resize', this._onResize);
    window.removeEventListener('mousemove', this._onMouseMove);

    // Restore canvas z-index
    this.canvas.style.zIndex = this._origZIndex;
    this.canvas.style.pointerEvents = '';
    this.canvas.style.opacity = '';

    // Restore splash
    const splash = document.getElementById('splashScreen');
    if (splash) {
      splash.style.background = '';
      splash.classList.remove('intro-active');
    }

    if (this.onComplete) this.onComplete();
  }

  destroy() {
    this.running = false;
    window.removeEventListener('resize', this._onResize);
    window.removeEventListener('mousemove', this._onMouseMove);
  }
}

function clamp(v, min, max) { return v < min ? min : v > max ? max : v; }

// ══════════════════════════════════════════
// INITIALIZE
// ══════════════════════════════════════════
let particleEngine = null;
let introEngine = null;

window.initParticles = function() {
  const canvas = document.getElementById('particleCanvas');
  if (!canvas) return;

  const enabled = localStorage.getItem('algochat_particles') !== 'false';
  if (!enabled) return;

  // Play geometric intro, then transition to particle engine
  introEngine = new GeoIntro(canvas, () => {
    introEngine = null;
    // Notify app that intro is done
    if (window._onIntroComplete) window._onIntroComplete();
    // Start normal particles
    if (!particleEngine) particleEngine = new ParticleEngine(canvas);
  });
};

// Auto-init when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', window.initParticles);
} else {
  window.initParticles();
}
