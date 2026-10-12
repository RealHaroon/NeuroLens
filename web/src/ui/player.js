// "Play forward pass": flies the camera station to station while the layers
// light up in order. Supports pause/resume, step back/forward and a speed factor.
//
// Every navigation bumps a token; long-running awaits check it and bail out, so
// stepping or restarting mid-animation never leaves two sequences running.

const DWELL_SECONDS = 2.4;
const FLY_SECONDS = 1.1;

export function createPlayer(scene, { onChange } = {}) {
  const stations = scene.stations;
  let index = -1; // -1 = overview, everything lit
  let playing = false;
  let speed = 1;
  let kernelSlide = false;
  let token = 0;
  let pausedMidRun = false; // a run() is suspended mid-sequence and can be resumed

  const state = () => ({ index, station: stations[index] ?? null, playing, atEnd: index === stations.length - 1 });
  const emit = () => onChange?.(state());

  /** Wait `seconds` of playback time; suspends while paused. Resolves false if cancelled. */
  function wait(seconds, my) {
    return new Promise((resolve) => {
      let left = seconds * 1000;
      let last = performance.now();
      const tick = (now) => {
        if (my !== token) return resolve(false);
        if (playing) left -= (now - last) * speed;
        last = now;
        if (left <= 0) return resolve(true);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  }

  /** Show station `i`: light it, fly there, run its animation. Resolves false if cancelled. */
  async function show(i, my) {
    index = i;
    scene.cancelAnimations();
    scene.setLit(i);
    emit();
    const id = stations[i];
    if (id === 'conv1' && kernelSlide) {
      await scene.focusRange(['input', 'conv1'], FLY_SECONDS / speed);
      if (my !== token) return false;
      await scene.playKernelSlide();
    } else {
      await scene.focus(id, FLY_SECONDS / speed);
      if (my !== token) return false;
      if (id === 'flatten') await scene.playFlatten();
    }
    return my === token;
  }

  async function run(from) {
    const my = ++token;
    for (let i = from; i < stations.length; i++) {
      if (!(await show(i, my))) return;
      if (!(await wait(DWELL_SECONDS, my))) return;
    }
    playing = false;
    pausedMidRun = false;
    scene.setLit(null); // pass complete: everything lit, stay on the output
    emit();
  }

  return {
    play() {
      if (playing) return;
      playing = true;
      scene.setPaused(false);
      if (pausedMidRun) {
        pausedMidRun = false;
        emit(); // the suspended run() picks up where it was
        return;
      }
      run(index >= 0 && index < stations.length - 1 ? index + 1 : 0);
    },
    pause() {
      if (!playing) return;
      playing = false;
      pausedMidRun = true;
      scene.setPaused(true);
      emit();
    },
    toggle() {
      if (playing) this.pause();
      else this.play();
    },
    async step(delta) {
      playing = false;
      pausedMidRun = false;
      scene.setPaused(false);
      const next = Math.max(0, Math.min(stations.length - 1, index + delta));
      const my = ++token;
      await show(next, my);
    },
    /** Jump to a station without playing (e.g. from the layer buttons). */
    goto(id) {
      playing = false;
      pausedMidRun = false;
      scene.setPaused(false);
      ++token;
      scene.cancelAnimations();
      index = stations.indexOf(id);
      scene.setLit(null);
      emit();
    },
    stop() {
      playing = false;
      pausedMidRun = false;
      scene.setPaused(false);
      ++token;
      scene.cancelAnimations();
      index = -1;
      scene.setLit(null);
      emit();
    },
    setSpeed(s) {
      speed = s;
      scene.setSpeed(s);
    },
    setKernelSlide(on) {
      kernelSlide = on;
    },
    state,
  };
}
