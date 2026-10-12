// 280x280 drawing surface: white ink on black, like MNIST. Pointer events cover
// mouse, pen and touch. Calls onChange() while drawing so predictions stay live.

export function createDrawpad(canvas, { onChange, onStart, lineWidth = 20 } = {}) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  let drawing = false;
  let last = null;

  function clear() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  function point(e) {
    const r = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * canvas.width,
      y: ((e.clientY - r.top) / r.height) * canvas.height,
    };
  }

  function stroke(from, to) {
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  }

  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    onStart?.();
    drawing = true;
    last = point(e);
    stroke(last, last);
    onChange?.();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drawing) return;
    const p = point(e);
    stroke(last, p);
    last = p;
    onChange?.();
  });
  const end = () => {
    drawing = false;
    last = null;
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);

  /** Show an uploaded image letterboxed on a neutral background (display only). */
  function showImage(img) {
    ctx.fillStyle = '#222';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const s = Math.min(canvas.width / img.width, canvas.height / img.height);
    const w = img.width * s;
    const h = img.height * s;
    ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
  }

  clear();
  return { clear, showImage, canvas };
}
