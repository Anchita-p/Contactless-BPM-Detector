/* PulseVision dashboard client.
   Telemetry arrives at 5 Hz over a WebSocket; the MJPEG feed is a plain <img>. */

const $ = (id) => document.getElementById(id);

const ALGORITHMS = [
  { key: 'pos',   name: 'POS',       tag: 'Main reading',
    blurb: 'Separates the pulse from your skin tone', color: '#6366F1' },
  { key: 'chrom', name: 'CHROM',     tag: 'Cross-check',
    blurb: 'Cancels glare using colour differences',  color: '#8B5CF6' },
  { key: 'green', name: 'Raw Green', tag: 'Simple baseline',
    blurb: 'Green light only, no correction',         color: '#10B981' },
];

const STRESS_STYLES = {
  Low:      ['bg-emerald-50', 'text-emerald-700', 'Low Stress'],
  Moderate: ['bg-amber-50',   'text-amber-700',   'Moderate Stress'],
  High:     ['bg-rose-50',    'text-rose-700',    'High Stress'],
};

// Reported SNR spans roughly 0-30 dB in practice; the confidence gate sits at
// 15x linear (~11.8 dB), so the meter is scaled to make that region readable.
const SNR_FULL_SCALE_DB = 30;

// Below this capture rate the machine is struggling, so the page stops
// spending anything on decoration and gives the CPU back to the pipeline.
const PERF_MODE_FPS = 14;

let socket = null;
let running = false;
let waveChart = null;
let evalChart = null;

/* ── Sidebar navigation ────────────────────────────────────────────────
   Each item names the view it belongs to and, optionally, a section to
   scroll to once that view is showing. */
const PAGE_TITLES = { live: 'Live Monitor', dataset: 'Accuracy Check' };

function showView(view) {
  $('view-live').classList.toggle('hidden', view !== 'live');
  $('view-dataset').classList.toggle('hidden', view !== 'dataset');
  $('page-title').textContent = PAGE_TITLES[view] || 'Live Monitor';
}

function closeSidebar() {
  $('sidebar').classList.add('-translate-x-full');
  $('scrim').classList.add('hidden');
}

document.querySelectorAll('.nav-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach((b) => b.classList.toggle('active', b === btn));
    showView(btn.dataset.view);
    closeSidebar();

    const target = btn.dataset.scroll;
    // Let the view become visible before measuring where to scroll to.
    requestAnimationFrame(() => {
      if (!target || target === 'top') { window.scrollTo({ top: 0, behavior: 'smooth' }); return; }
      $(target)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
});

$('menu').addEventListener('click', () => {
  $('sidebar').classList.remove('-translate-x-full');
  $('scrim').classList.remove('hidden');
});
$('scrim').addEventListener('click', closeSidebar);

/* ── Algorithm cards ───────────────────────────────────────────────────── */
function buildAlgorithmCards() {
  $('algo-grid').innerHTML = ALGORITHMS.map((a) => `
    <div class="rounded-2xl border border-slate-100 bg-slate-50/60 p-5">
      <div class="flex items-start justify-between gap-2">
        <div>
          <div class="font-bold" style="color:${a.color}">${a.name}</div>
          <div class="text-[11px] text-ink-400">${a.blurb}</div>
        </div>
        <span class="text-[10px] font-bold px-2 py-1 rounded-full bg-white border border-slate-100 text-ink-600 whitespace-nowrap">${a.tag}</span>
      </div>
      <div class="mt-3 flex items-end gap-1.5">
        <span id="algo-${a.key}" class="text-4xl font-extrabold tabular-nums">--</span>
        <span class="text-sm font-bold text-ink-400 mb-1">BPM</span>
      </div>
      <div id="algo-${a.key}-note" class="mt-1 text-[11px] text-ink-400">Waiting for a clear pulse…</div>
    </div>`).join('');
}

/* ── Charts ────────────────────────────────────────────────────────────── */
function initWaveChart() {
  waveChart = new Chart($('waveChart'), {
    type: 'line',
    data: { labels: [], datasets: [{
      data: [], borderColor: '#6366F1', borderWidth: 2, tension: 0.35,
      pointRadius: 0, fill: true,
      backgroundColor: (ctx) => {
        const { chart } = ctx;
        if (!chart.chartArea) return 'rgba(99,102,241,0.10)';
        const g = chart.ctx.createLinearGradient(0, chart.chartArea.top, 0, chart.chartArea.bottom);
        g.addColorStop(0, 'rgba(99,102,241,0.28)');
        g.addColorStop(1, 'rgba(99,102,241,0.00)');
        return g;
      },
    }] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false, resizeDelay: 100,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { display: false },
        y: { min: -1.15, max: 1.15, grid: { color: '#F1F5F9' }, ticks: { display: false } },
      },
    },
  });
}

// Created lazily, not at boot. A Chart.js canvas constructed inside a
// display:none parent measures a zero-height box and renders at the wrong
// scale forever after; by first render the dataset tab is visible.
function ensureEvalChart() {
  if (evalChart) return evalChart;
  evalChart = new Chart($('evalChart'), {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true, maintainAspectRatio: false, resizeDelay: 100,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 8, font: { weight: '600' } } } },
      scales: {
        x: { title: { display: true, text: 'Time (s)' }, grid: { display: false } },
        y: { title: { display: true, text: 'BPM' }, grid: { color: '#F1F5F9' } },
      },
    },
  });
  return evalChart;
}

/* ── Live telemetry rendering ──────────────────────────────────────────── */
function setBanner(text, tone) {
  const banner = $('banner');
  if (!text) { banner.classList.add('hidden'); return; }
  const tones = {
    error: 'bg-rose-50 border-rose-200 text-rose-700',
    warn:  'bg-amber-50 border-amber-200 text-amber-800',
    info:  'bg-brand-50 border-brand-100 text-brand-700',
  };
  banner.className = `mb-5 rounded-xl border px-4 py-3 text-sm font-medium ${tones[tone] || tones.info}`;
  banner.textContent = text;
}

const STATE_LOOK = {
  TRACKING:    ['bg-emerald-400', 'Reading your pulse'],
  CALIBRATING: ['bg-amber-400',   'Getting ready'],
  FACE_LOST:   ['bg-rose-400',    "Can't see you"],
  IDLE:        ['bg-slate-300',   'Camera off'],
};

const VIDEO_HINTS = {
  TRACKING:    'Looking good — stay comfortable.',
  CALIBRATING: 'Hold still and look at the camera.',
  FACE_LOST:   'Move back into frame so we can find you again.',
  IDLE:        'Sit comfortably and look at the camera.',
};

function setStateChip(state) {
  const [dot, label] = STATE_LOOK[state] || STATE_LOOK.IDLE;
  const dotEl = $('state-dot');
  dotEl.className = `w-1.5 h-1.5 rounded-full ${dot}` + (state === 'TRACKING' ? ' livedot' : '');
  $('state-text').textContent = label;
  $('video-hint').textContent = VIDEO_HINTS[state] || VIDEO_HINTS.IDLE;
}

function renderTelemetry(t) {
  if (!t) return;

  setStateChip(t.quality.state);

  // Header pills
  $('pill-camera').classList.toggle('hidden', !running);
  $('pill-camera').classList.toggle('inline-flex', running);
  $('pill-fps').classList.toggle('hidden', t.quality.fps === null);
  $('pill-fps').classList.toggle('inline-flex', t.quality.fps !== null);
  if (t.quality.fps !== null) {
    const fps = `${Math.round(t.quality.fps)} fps`;
    $('pill-fps').textContent = fps;
    $('fps-chip').textContent = fps;
    document.body.classList.toggle('perf', t.quality.fps < PERF_MODE_FPS);
  }

  // Calibration
  const calibrating = t.calibration_progress < 1;
  const pct = Math.round(t.calibration_progress * 100);
  $('calibration').classList.toggle('hidden', !calibrating);
  $('calib-bar').style.width = `${pct}%`;
  $('calib-pct').textContent = `${pct}%`;
  const left = Math.max(0, Math.ceil(t.calibration_target_seconds - t.calibration_seconds));
  $('calib-sub').textContent = left > 0
    ? `About ${left} second${left === 1 ? '' : 's'} to go — hold still and keep your face in frame.`
    : 'Buffer full — locking on to your pulse…';

  const calibPill = $('pill-calib');
  calibPill.classList.toggle('hidden', !calibrating);
  calibPill.classList.toggle('inline-flex', calibrating);
  if (calibrating) calibPill.textContent = `Warming up ${pct}%`;

  if (t.quality.state === 'FACE_LOST') {
    setBanner("We've lost sight of your face. Move back into frame — we'll pick up where we left off.", 'error');
  } else if (calibrating) {
    const left = Math.max(0, Math.ceil(t.calibration_target_seconds - t.calibration_seconds));
    setBanner(`Collecting heartbeats — about ${left} second${left === 1 ? '' : 's'} to go. `
      + 'We need a full 13 seconds of video before a reading means anything.', 'info');
  } else if (t.bpm === null) {
    // The bar fills when the buffer is full, but the first window still has to
    // clear the confidence gate. Saying nothing here reads as the app hanging.
    setBanner('Buffer full — locking on to your pulse. This usually takes a few more seconds.', 'info');
  } else {
    setBanner(null);
  }

  // Vitals visibility
  const unlocked = !calibrating;
  $('vitals').classList.toggle('hidden', !unlocked);
  $('vitals-locked').classList.toggle('hidden', unlocked);

  // Heart rate
  $('bpm').textContent = t.bpm === null ? '--' : Math.round(t.bpm);
  // The displayed value is a moving average of recent windows. When the
  // current window fails its own confidence gate the number is a held-over
  // reading, not a live one, and saying so beats quietly implying otherwise.
  const stale = t.bpm !== null && t.algorithms.pos.bpm === null;
  $('hr-icon').classList.toggle('beating', t.bpm !== null && !stale);
  $('bpm-stale').classList.toggle('hidden', !stale);
  const range = $('bpm-range');
  if (t.bpm === null) {
    range.textContent = 'Typical resting range is 60–100';
    range.className = 'mt-3 inline-block text-xs font-bold px-3 py-1.5 rounded-full bg-slate-100 text-ink-600';
  } else {
    const inRange = t.bpm >= 60 && t.bpm <= 100;
    range.textContent = inRange
      ? 'In the typical resting range'
      : (t.bpm < 60 ? 'Below the typical resting range' : 'Above the typical resting range');
    range.className = `mt-3 inline-block text-xs font-bold px-3 py-1.5 rounded-full ${
      inRange ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'}`;
  }

  // Stress
  $('rmssd').textContent = t.stress.rmssd === null ? '--' : t.stress.rmssd.toFixed(1);
  const badge = $('stress-badge');
  const style = STRESS_STYLES[t.stress.tier];
  if (style) {
    badge.textContent = style[2];
    badge.className = `mt-3 inline-block text-xs font-bold px-3 py-1.5 rounded-full ${style[0]} ${style[1]}`;
  } else {
    badge.textContent = t.stress.n_beats
      ? `Counting heartbeats… ${t.stress.n_beats} so far`
      : 'Waiting for a steady pulse';
    badge.className = 'mt-3 inline-block text-xs font-bold px-3 py-1.5 rounded-full bg-slate-100 text-ink-600';
  }

  // Signal confidence
  const snr = t.quality.snr_db;
  $('snr-value').textContent = snr === null ? '-- dB' : `${snr.toFixed(1)} dB`;
  $('snr-bar').style.width = snr === null ? '0%'
    : `${Math.max(0, Math.min(100, (snr / SNR_FULL_SCALE_DB) * 100))}%`;

  // Algorithms
  ALGORITHMS.forEach((a) => {
    const entry = t.algorithms[a.key];
    $(`algo-${a.key}`).textContent = entry.bpm === null ? '--' : Math.round(entry.bpm);
    $(`algo-${a.key}-note`).textContent = entry.bpm === null
      ? 'No clear pulse yet'
      : `Signal quality ${entry.snr_db === null ? '--' : entry.snr_db.toFixed(1)} dB`;
  });

  const ag = $('agreement');
  if (ag) {
    const reporting = t.agreement.reporting;
    if (t.agreement.spread_bpm === null) {
      ag.textContent = reporting
        ? `Only 1 of 3 methods can see a clear pulse right now — try better lighting.`
        : 'No method can see a clear pulse yet.';
      ag.className = 'mt-4 rounded-xl px-4 py-3 text-sm font-semibold bg-slate-50 text-ink-600';
    } else if (t.agreement.consistent) {
      // Say how many actually reported. Claiming "all three" when one found no
      // pulse overstates the confidence the reading deserves.
      ag.textContent = reporting === 3
        ? `All three methods agree to within ${t.agreement.spread_bpm} BPM — this reading is solid.`
        : `${reporting} of 3 methods agree to within ${t.agreement.spread_bpm} BPM. `
          + 'The third has no clear pulse, so treat this as good but not confirmed.';
      ag.className = reporting === 3
        ? 'mt-4 rounded-xl px-4 py-3 text-sm font-semibold bg-emerald-50 text-emerald-700'
        : 'mt-4 rounded-xl px-4 py-3 text-sm font-semibold bg-brand-50 text-brand-700';
    } else {
      ag.textContent = `The methods differ by ${t.agreement.spread_bpm} BPM — usually a sign of movement. Try holding still.`;
      ag.className = 'mt-4 rounded-xl px-4 py-3 text-sm font-semibold bg-amber-50 text-amber-800';
    }
  }

  // Waveform
  if (waveChart && t.waveform.length) {
    waveChart.data.labels = t.waveform.map((_, i) => i);
    waveChart.data.datasets[0].data = t.waveform;
    waveChart.update('none');
  }
}

/* ── Start / stop ──────────────────────────────────────────────────────── */
function setToggle(on) {
  running = on;
  $('toggle-text').textContent = on ? 'Stop' : 'Start';
  $('toggle').className = `mt-4 w-full py-3 rounded-xl font-bold text-white transition flex items-center justify-center gap-2 ${
    on ? 'bg-rose-500 hover:bg-rose-600' : 'bg-brand-600 hover:bg-brand-700'}`;
  $('video').classList.toggle('hidden', !on || $('hide-preview').checked);
  $('video-overlay').classList.toggle('hidden', !on);
  $('video-placeholder').classList.toggle('hidden', on && !$('hide-preview').checked);
  $('preview-off').classList.toggle('hidden', !(on && $('hide-preview').checked));
  if (!on) {
    setStateChip('IDLE');
    $('video').removeAttribute('src');
    ['pill-camera', 'pill-calib', 'pill-fps'].forEach((id) => {
      $(id).classList.add('hidden'); $(id).classList.remove('inline-flex');
    });
    $('vitals').classList.add('hidden');
    $('vitals-locked').classList.remove('hidden');
    $('calibration').classList.add('hidden');
  }
}

/* Dropping the <img> src closes the MJPEG response, which drops the server's
   viewer count to zero and stops it encoding JPEGs at all. On a busy machine
   that is the single biggest saving available without stopping the reading. */
function applyPreviewVisibility() {
  const hidden = $('hide-preview').checked;
  $('video').classList.toggle('hidden', hidden || !running);
  $('video-overlay').classList.toggle('hidden', !running);
  $('video-placeholder').classList.toggle('hidden', running && !hidden);
  $('preview-off').classList.toggle('hidden', !(running && hidden));

  if (!running) return;
  if (hidden) {
    $('video').removeAttribute('src');
  } else if (!$('video').getAttribute('src')) {
    $('video').src = `/video_feed?t=${Date.now()}`;
  }
}

$('hide-preview').addEventListener('change', applyPreviewVisibility);

function openSocket() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
  socket.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.error) { setBanner(msg.error, 'error'); return; }
    // Once the server says capture has stopped, leave the idle UI alone --
    // a late frame must not repaint stale vitals beside a dead camera.
    if (!msg.running) { if (running) setToggle(false); return; }
    renderTelemetry(msg.telemetry);
  };
  socket.onclose = () => { socket = null; };
}

$('toggle').addEventListener('click', async () => {
  if (running) {
    await fetch('/api/stop', { method: 'POST' });
    if (socket) socket.close();
    setToggle(false);
    setBanner(null);
    return;
  }

  setBanner('Starting camera…', 'info');
  const res = await fetch('/api/start', { method: 'POST' });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    setBanner(body.detail || 'Could not start the camera.', 'error');
    return;
  }
  setToggle(true);
  applyPreviewVisibility();
  openSocket();
});

/* ── Dataset evaluation ────────────────────────────────────────────────── */
function wireDropzone(zoneId, inputId, labelId, placeholder) {
  const zone = $(zoneId);
  const input = $(inputId);
  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragging'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragging');
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      input.dispatchEvent(new Event('change'));
    }
  });
  input.addEventListener('change', () => {
    $(labelId).textContent = input.files.length ? input.files[0].name : placeholder;
    $('run-eval').disabled = !$('file-video').files.length;
  });
}

function evalStatus(text, tone) {
  const el = $('eval-status');
  el.classList.toggle('hidden', !text);
  const tones = { error: 'text-rose-600', busy: 'text-brand-700', ok: 'text-emerald-600' };
  el.className = `mt-4 text-sm font-medium ${tones[tone] || 'text-ink-600'}`;
  el.textContent = text || '';
}

async function runEvaluation(request, note) {
  $('run-eval').disabled = true;
  $('run-sample').disabled = true;
  evalStatus(note, 'busy');

  try {
    const res = await fetch(request.url, { method: 'POST', body: request.body });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || 'That did not work.');
    renderEvaluation(body);
    const scored = body.ground_truth
      ? ''
      : " No heart-rate file was included, so there's nothing to score against — these are our readings only.";
    evalStatus(`Done — ${body.frames_evaluated} readings from this video.${scored}`, 'ok');
  } catch (err) {
    evalStatus(err.message, 'error');
  } finally {
    $('run-eval').disabled = !$('file-video').files.length;
    $('run-sample').disabled = false;
  }
}

$('run-eval').addEventListener('click', () => {
  const video = $('file-video').files[0];
  if (!video) return;
  const form = new FormData();
  form.append('video', video);
  if ($('file-gt').files[0]) form.append('ground_truth', $('file-gt').files[0]);
  runEvaluation({ url: '/api/analyze_dataset', body: form },
    'Working through the video frame by frame — a long clip can take a few minutes…');
});

$('run-sample').addEventListener('click', () => {
  runEvaluation({ url: '/api/analyze_sample' },
    'Replaying the bundled sample clip — about half a minute…');
});

function renderEvaluation(data) {
  $('eval-results').classList.remove('hidden');
  const chart = ensureEvalChart();

  const pos = data.algorithms.pos;
  $('m-mae').textContent  = pos.metrics.mae  === null ? '--' : pos.metrics.mae;
  $('m-rmse').textContent = pos.metrics.rmse === null ? '--' : pos.metrics.rmse;
  $('m-snr').textContent  = data.mean_snr_db === null ? '--' : data.mean_snr_db;

  $('gt-note').textContent = data.ground_truth
    ? `We read your pulse file as: ${data.ground_truth_note}`
    : "No true-pulse file was uploaded, so there's nothing to compare against yet — showing our readings only.";

  const datasets = ALGORITHMS.map((a) => ({
    label: a.name,
    data: data.algorithms[a.key].bpm,
    borderColor: a.color,
    backgroundColor: a.color,
    borderWidth: 2, pointRadius: 0, tension: 0.3, spanGaps: true,
  }));
  if (data.ground_truth) {
    datasets.push({
      label: 'Ground truth',
      data: data.ground_truth,
      borderColor: '#475569', backgroundColor: '#475569',
      borderWidth: 2.5, borderDash: [6, 4], pointRadius: 0, tension: 0.3, spanGaps: true,
    });
  }
  chart.data.labels = data.timestamps;
  chart.data.datasets = datasets;
  chart.update();

  $('eval-table').innerHTML = ALGORITHMS.map((a) => {
    const m = data.algorithms[a.key].metrics;
    const cell = (v, suffix = '') =>
      (v === null || v === undefined ? '<span class="text-ink-400">--</span>' : `${v}${suffix}`);
    const loa = m.loa ? `${m.loa[0]} to ${m.loa[1]}` : null;
    return `<tr>
      <td class="py-3">
        <div class="font-bold" style="color:${a.color}">${a.name}</div>
        <div class="text-[11px] text-ink-400">${a.blurb}</div>
      </td>
      <td class="py-3 text-right tabular-nums font-semibold">${cell(m.mae)}</td>
      <td class="py-3 text-right tabular-nums font-semibold">${cell(m.rmse)}</td>
      <td class="py-3 text-right tabular-nums text-ink-600">${cell(m.mse)}</td>
      <td class="py-3 text-right tabular-nums text-ink-600">${cell(m.mape, '%')}</td>
      <td class="py-3 text-right tabular-nums text-ink-600">${cell(m.r)}</td>
      <td class="py-3 text-right tabular-nums text-ink-600">${cell(m.bias)}</td>
      <td class="py-3 text-right tabular-nums text-ink-600 whitespace-nowrap">${cell(loa)}</td>
      <td class="py-3 text-right tabular-nums text-ink-600">${cell(m.within_5, '%')}</td>
      <td class="py-3 text-right tabular-nums text-ink-400">${m.n}</td>
    </tr>`;
  }).join('');
}

/* ── Boot ──────────────────────────────────────────────────────────────── */
buildAlgorithmCards();
initWaveChart();
wireDropzone('drop-video', 'file-video', 'name-video', 'or click to browse');
wireDropzone('drop-gt', 'file-gt', 'name-gt', 'UBFC-rPPG / PURE style accepted');
$('run-eval').disabled = true;
setToggle(false);
showView('live');
