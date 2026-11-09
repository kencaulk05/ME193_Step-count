// How many steps? live page. Talks to main.py through socket.io: it sends 'cmd' and receives 'state' and 'notice'.
(function () {
  const socket = io(window.location.origin);
  const $ = (id) => document.getElementById(id);
  const store = {
    get(k, d) { try { const v = window.localStorage.getItem('live.' + k); return v === null ? d : v; } catch (e) { return d; } },
    set(k, v) { try { window.localStorage.setItem('live.' + k, v); } catch (e) { /* private mode: just do not remember */ } }
  };
  let mode = 'idle', imuOk = false, lastId = null, formFor = null;

  $('person').value = store.get('person', '');
  $('person').addEventListener('input', () => { store.set('person', $('person').value); updateButtons(); });

  function updateButtons() {
    const idle = mode !== 'walking';
    $('start').disabled = !(idle && imuOk && $('person').value.trim());
    $('stop').disabled = idle;
    $('person').disabled = !idle;
  }

  $('start').addEventListener('click', () => socket.emit('cmd', { action: 'start', person: $('person').value.trim() }));
  $('stop').addEventListener('click', () => socket.emit('cmd', { action: 'stop' }));
  $('save').addEventListener('click', () => { if (lastId) socket.emit('cmd', { action: 'truth', id: lastId, steps: $('truth').value.trim(), distance: $('truthDist').value.trim() }); });
  $('discard').addEventListener('click', () => { if (lastId && window.confirm('Delete this walk?')) socket.emit('cmd', { action: 'delete', id: lastId }); });
  $('again').addEventListener('click', () => socket.emit('cmd', { action: 'reset' }));

  socket.on('connect', () => { $('conn').textContent = 'connected'; $('conn').className = 'chip good'; socket.emit('cmd', { action: 'refresh' }); });
  socket.on('disconnect', () => { $('conn').textContent = 'disconnected'; $('conn').className = 'chip bad'; });
  socket.on('notice', (n) => { const el = $('notice'); el.className = 'notice ' + (n.level || 'info'); el.textContent = n.text; });
  socket.on('state', render);

  const fmt = (x, d) => (x === null || x === undefined || Number.isNaN(x)) ? '–' : Number(x).toFixed(d);
  function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
  const plural = (n, w) => n + ' ' + w + (n === 1 ? '' : 's');

  function render(s) {
    mode = s.mode; imuOk = !!s.imu_ok;
    $('imu').textContent = 'IMU: ' + s.imu_rate + '/s';
    $('imu').className = 'chip ' + (imuOk ? 'good' : 'bad');
    const walking = s.mode === 'walking', done = s.mode === 'done';
    $('count').textContent = String(s.steps);
    $('count').className = 'count' + (walking ? ' live' : done ? ' done' : '');
    $('pace').textContent = walking && s.pace ? String(s.pace) : '–';
    $('dist').textContent = s.distance_m === null || s.distance_m === undefined ? '–' : fmt(s.distance_m, 1);
    $('distlab').textContent = s.step_length_m ? 'metres (' + fmt(s.step_length_m * 100, 0) + ' cm a step)' : 'metres';
    $('time').textContent = walking || done ? fmt(s.elapsed, 0) : '–';
    if (walking) $('sub').textContent = s.person + ' is walking. The last half second of the count is an estimate. Press STOP when you are done.';
    else if (done) $('sub').textContent = 'Final count for ' + s.person + ': exactly what the network says for the whole walk.';
    else $('sub').textContent = imuOk ? 'Type your name, press START and walk.' : 'No IMU data: check the sensor and the board.';
    const warn = $('warn');
    if (!imuOk && walking) { warn.hidden = false; warn.textContent = 'The sensor stopped sending readings: the count is paused.'; } else warn.hidden = true;
    if (s.step_length_source && (walking || done) && s.distance_m !== null && s.distance_m !== undefined) $('distlab').title = 'step length from ' + s.step_length_source;
    updateButtons();
    renderPeople(s.people || []);
    renderResult(s);
    renderTally(s);
    renderEval(s.evaluation || {});
    drawPlot((s.plot && s.plot.density) || [], (s.plot && s.plot.gyro) || []);
  }

  function renderPeople(people) {
    const dl = $('people');
    if (dl.children.length === people.length) return;
    dl.textContent = '';
    people.forEach((p) => { const o = document.createElement('option'); o.value = p; dl.appendChild(o); });
  }

  function renderResult(s) {
    const box = $('result');
    const last = s.last;
    if (s.mode !== 'done' || !last) { box.hidden = true; formFor = null; return; }
    box.hidden = false;
    lastId = last.id;
    $('resultText').textContent = 'The counter says ' + plural(last.steps, 'step') + ' (exactly ' + fmt(last.steps_exact, 2) + ') in ' + fmt(last.seconds, 0) + ' seconds.';
    $('distText').textContent = last.distance_est_m === '' || last.distance_est_m === null ? 'No distance estimate: there is no step-length model.' :
      'About ' + fmt(last.distance_est_m, 1) + ' m, at ' + fmt(last.step_length_m * 100, 0) + ' cm a step, from ' + (last.step_length_source || 'a step-length model') + '.';
    const answered = last.true_steps !== '' && last.true_steps !== null && last.true_steps !== undefined;
    $('truthForm').hidden = answered;
    $('truthDone').hidden = !answered;
    $('again').hidden = !answered;
    if (answered) {
      const e = last.error;
      $('truthDone').textContent = 'You counted ' + plural(last.true_steps, 'step') + ': the counter was ' + (e === 0 ? 'exactly right.' : plural(Math.abs(e), 'step') + (e > 0 ? ' too many.' : ' too few.'));
    } else if (formFor !== last.id) {
      formFor = last.id;
      $('truth').value = ''; $('truthDist').value = '';
    }
  }

  function renderTally(s) {
    const t = s.tally || { n: 0 };
    $('tallyText').textContent = t.n ? plural(t.n, 'demo') + ' with a counted answer: on average ' + fmt(t.mae, 2) + ' steps off' + (t.mape !== null && t.mape !== undefined ? ' (' + fmt(t.mape, 1) + '% of the count)' : '') +
      ', ' + fmt(100 * t.within1, 0) + '% within one step, worst ' + fmt(t.worst, 0) + '.' : 'No demos with a counted answer yet.';
    const body = document.querySelector('#sessions tbody');
    body.textContent = '';
    const rows = s.sessions || [];
    if (!rows.length) return;
    const head = el('tr');
    [['when', ''], ['who', ''], ['counter', 'num'], ['you', 'num'], ['off by', 'num'], ['metres', 'num']].forEach((h) => head.appendChild(el('th', h[1], h[0])));
    body.appendChild(head);
    rows.forEach((r) => {
      const tr = el('tr');
      const answered = r.true_steps !== '' && r.true_steps !== null && r.true_steps !== undefined;
      [[String(r.start).slice(11, 16), ''], [r.person, ''], [String(r.steps), 'num'], [answered ? String(r.true_steps) : '?', 'num'], [answered ? (r.error > 0 ? '+' : '') + r.error : '', 'num'],
       [r.distance_est_m === '' || r.distance_est_m === null ? '' : fmt(r.distance_est_m, 1), 'num']].forEach((c) => tr.appendChild(el('td', c[1], c[0])));
      body.appendChild(tr);
    });
  }

  function renderEval(e) {
    const box = $('evalText');
    if (!e.n_trials) { box.textContent = 'No evaluation numbers were found next to the network.'; return; }
    const ci = e.network_mae_ci95 || [null, null];
    const people = e.distance_per_person_mape || {};
    const ppl = Object.keys(people).map((k) => 'person ' + k + ' ' + fmt(people[k], 0) + '%').join(', ');
    box.textContent = (e.synthetic ? 'MADE-UP TRIALS: not real accuracy. ' : '') + 'On ' + e.n_trials + ' trials it had never seen: ' + fmt(e.network_mae, 2) + ' steps off on average (95% interval ' + fmt(ci[0], 2) + ' to ' + fmt(ci[1], 2) +
      '), ' + fmt(e.network_mae_rounded, 2) + ' once rounded to a whole number as on this screen, worst ' + fmt(e.worst, 1) + '. It counted ' + fmt(e.false_steps_network, 1) + ' steps while standing or tapping a foot. ' +
      'Distance is steps times a step length: about ' + fmt(e.distance_end_to_end_mape, 0) + '% off with one model for everyone' + (ppl ? ' (' + ppl + ', each calibrated on that person alone)' : '') + '. The truth in those trials was a human count, good to a step or two, and the walks were only 5 to 6 metres.';
  }

  function drawPlot(dens, gyro) {
    const canvas = $('plot');
    const ctx = canvas.getContext ? canvas.getContext('2d') : null;
    if (!ctx) return;
    const ratio = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 300, h = 130;
    canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const style = getComputedStyle(document.documentElement);
    const col = (name, d) => (style.getPropertyValue(name) || d).trim();
    function line(values, top, color, fill) {
      if (values.length < 2) return;
      ctx.beginPath();
      values.forEach((v, i) => { const x = (i / 150) * w, y = h - 4 - (Math.min(v, top) / top) * (h - 8); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      if (fill) { ctx.lineTo(((values.length - 1) / 150) * w, h - 4); ctx.lineTo(0, h - 4); ctx.closePath(); ctx.fillStyle = fill; ctx.globalAlpha = 0.3; ctx.fill(); ctx.globalAlpha = 1; ctx.beginPath(); values.forEach((v, i) => { const x = (i / 150) * w, y = h - 4 - (Math.min(v, top) / top) * (h - 8); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); }
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
    }
    line(gyro, Math.max(400, ...gyro), col('--gyro', '#d1862b'));
    line(dens, Math.max(4, ...dens), col('--dens', '#2ca02c'), col('--dens', '#2ca02c'));
  }

  updateButtons();
})();
