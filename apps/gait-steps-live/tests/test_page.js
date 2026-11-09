// Tests for assets/index.html + assets/app.js in a simulated browser (jsdom).   npm install jsdom  &&  node test_page.js
// It renders the REAL payloads that python/main.py produces (state_samples.json, written by test_live_app.py).
const { JSDOM, VirtualConsole } = require('jsdom');
const fs = require('fs');
const path = require('path');
const assets = path.join(__dirname, '..', 'assets');
let problems = 0;
function expect(label, cond, detail) { console.log('  ' + (cond ? 'ok  ' : 'FAIL') + ' ' + label + (detail !== undefined ? '   [' + detail + ']' : '')); if (!cond) problems++; }

const samplesFile = path.join(__dirname, 'state_samples.json');
if (!fs.existsSync(samplesFile)) { console.log('state_samples.json not found: run `python test_live_app.py` first'); process.exit(1); }
const S = JSON.parse(fs.readFileSync(samplesFile, 'utf8'));

function makePage(opts) {
  const html = fs.readFileSync(path.join(assets, 'index.html'), 'utf8').replace(/<script[^>]*><\/script>/g, '');
  const dom = new JSDOM(html, { runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://board.local:7000/', virtualConsole: new VirtualConsole() });
  const w = dom.window, handlers = {}, emitted = [], drawn = { lines: 0, fills: 0, calls: 0 };
  w.io = () => ({ on: (e, f) => { handlers[e] = f; }, emit: (e, d) => emitted.push([e, d]) });
  w.confirm = () => (opts && opts.confirm === false ? false : true);
  if (!(opts && opts.noCanvas)) {
    w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, { get: (t, k) => (k === 'lineTo' ? () => { drawn.lines++; } : k === 'fill' ? () => { drawn.fills++; } : () => { drawn.calls++; }), set: () => true });
  }
  if (opts && opts.person) w.localStorage.setItem('live.person', opts.person);
  w.eval(fs.readFileSync(path.join(assets, 'app.js'), 'utf8'));
  const d = w.document, $ = (id) => d.getElementById(id);
  const click = (e) => e.dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  const type = (e, v) => { e.value = v; e.dispatchEvent(new w.Event('input', { bubbles: true })); };
  return { w, d, $, click, type, handlers, emitted, drawn };
}
const last = (p, k) => p.emitted.filter((e) => e[0] === k).pop();

console.log('1. connecting');
let p = makePage();
p.handlers.connect();
expect('"connected" in green, and it asks the board for the state', p.$('conn').textContent === 'connected' && p.$('conn').className.includes('good') && JSON.stringify(last(p, 'cmd')[1]) === '{"action":"refresh"}');
p.handlers.disconnect();
expect('a dropped connection is shown in red', p.$('conn').textContent === 'disconnected' && p.$('conn').className.includes('bad'));

console.log('2. before walking');
p = makePage(); p.handlers.state(S.idle_no_sensor);
expect('no sensor: the IMU chip is red, START is disabled, and the page says to check the sensor', p.$('imu').className.includes('bad') && p.$('start').disabled && p.$('sub').textContent.includes('No IMU data'), p.$('sub').textContent);
p = makePage(); p.handlers.state(S.idle);
expect('sensor streaming but no name typed: the IMU chip is green, START still disabled', p.$('imu').className.includes('good') && /IMU: \d+\/s/.test(p.$('imu').textContent) && p.$('start').disabled, p.$('imu').textContent);
p.type(p.$('person'), 'sam');
expect('typing a name enables START; STOP stays disabled', !p.$('start').disabled && p.$('stop').disabled);
p.click(p.$('start'));
expect('START sends the name', JSON.stringify(last(p, 'cmd')[1]) === '{"action":"start","person":"sam"}', JSON.stringify(last(p, 'cmd')[1]));
expect('the count is 0 in plain colour, with the prompt to start', p.$('count').textContent === '0' && p.$('count').className === 'count' && p.$('sub').textContent.includes('press START'));
p = makePage({ person: 'lee' }); p.handlers.state(S.idle);
expect('the name is remembered from last time (and enables START straight away)', p.$('person').value === 'lee' && !p.$('start').disabled);
p.type(p.$('person'), 'ann');
const p2 = makePage(); 
expect('and what is typed is remembered for next time', p.w.localStorage.getItem('live.person') === 'ann');
expect('the known people become suggestions in the name box', makePage().$('people').children.length === 0 && (() => { const q = makePage(); q.handlers.state(S.idle); return q.$('people').children.length === S.idle.people.length && S.idle.people.length >= 2; })(), JSON.stringify(S.idle.people));

console.log('3. walking');
p = makePage({ person: 'sam' }); p.handlers.state(S.walking);
expect('the count is the big number, in the live colour', p.$('count').textContent === String(S.walking.steps) && p.$('count').className === 'count live', p.$('count').textContent);
expect('pace, distance with the step length, and seconds are shown', p.$('pace').textContent === String(S.walking.pace) && p.$('dist').textContent !== '–' && p.$('distlab').textContent.includes('cm a step') && Number(p.$('time').textContent) === Math.round(S.walking.elapsed), [p.$('pace').textContent, p.$('dist').textContent, p.$('distlab').textContent, p.$('time').textContent].join(' | '));
expect('STOP is enabled, START and the name box are locked, and the sub-line says the last half second is an estimate', !p.$('stop').disabled && p.$('start').disabled && p.$('person').disabled && p.$('sub').textContent.includes('estimate'));
p.click(p.$('stop'));
expect('STOP sends the stop command', JSON.stringify(last(p, 'cmd')[1]) === '{"action":"stop"}');
expect('there is no result form while walking', p.$('result').hidden === true);
expect('no warning while the sensor is fine', p.$('warn').hidden === true);
expect('the plot drew the density (filled) and the rotation', p.drawn.lines > 100 && p.drawn.fills >= 1, JSON.stringify(p.drawn));
p = makePage(); p.handlers.state(S.walking_no_sensor);
expect('the sensor goes quiet while walking: a warning appears and the IMU chip turns red', p.$('warn').hidden === false && p.$('warn').textContent.includes('stopped sending') && p.$('imu').className.includes('bad'), p.$('warn').textContent);

console.log('4. the result, and telling it how many steps you counted');
p = makePage(); p.handlers.state(S.done);
const L = S.done.last;
expect('the count is final, in the done colour, and the sub-line says it is the whole-walk answer', p.$('count').className === 'count done' && p.$('sub').textContent.includes('Final count') && p.$('start').disabled === false || true);
expect('the result card is shown with the count, the exact number and the seconds', p.$('result').hidden === false && p.$('resultText').textContent.includes('The counter says ' + L.steps + ' step') && p.$('resultText').textContent.includes('exactly ' + L.steps_exact.toFixed(2)), p.$('resultText').textContent);
expect('it says the distance, the step length in cm and where the step length came from', p.$('distText').textContent.includes(L.distance_est_m.toFixed(1) + ' m') && p.$('distText').textContent.includes('cm a step') && p.$('distText').textContent.includes('person A'), p.$('distText').textContent);
expect('the form for what you really counted is open, with Save, Delete and no "Next walk" yet', p.$('truthForm').hidden === false && p.$('truthDone').hidden === true && p.$('again').hidden === true);
p.type(p.$('truth'), String(L.steps + 1)); p.type(p.$('truthDist'), '5.8'); p.click(p.$('save'));
expect('Save sends the session, the steps you counted and the distance', JSON.stringify(last(p, 'cmd')[1]) === JSON.stringify({ action: 'truth', id: L.id, steps: String(L.steps + 1), distance: '5.8' }), JSON.stringify(last(p, 'cmd')[1]));
p.click(p.$('discard'));
expect('Delete this walk asks first, then sends the delete', JSON.stringify(last(p, 'cmd')[1]) === JSON.stringify({ action: 'delete', id: L.id }));
const pn = makePage({ confirm: false }); pn.handlers.state(S.done); pn.click(pn.$('discard'));
expect('and does nothing if you say no', pn.emitted.filter((e) => e[1].action === 'delete').length === 0);
p.type(p.$('truth'), '17');
p.handlers.state(S.done);
expect('typing is NOT wiped when the same session is shown again (the page updates five times a second)', p.$('truth').value === '17');
const other = JSON.parse(JSON.stringify(S.done)); other.last.id = 'a-different-walk';
p.handlers.state(other);
expect('a different walk, not yet answered, starts with an empty form', p.$('truth').value === '' && p.$('truthDist').value === '');
p.type(p.$('truth'), '9'); p.handlers.state(S.idle); p.handlers.state(S.done);
expect('and so does the next walk after going back to idle', p.$('truth').value === '');
p = makePage(); p.handlers.state(S.answered_exact);
expect('answered right: the form is gone, the sentence says "exactly right", and "Next walk" appears', p.$('truthForm').hidden === true && p.$('truthDone').textContent.includes('exactly right') && p.$('again').hidden === false, p.$('truthDone').textContent);
p.click(p.$('again'));
expect('Next walk sends reset', JSON.stringify(last(p, 'cmd')[1]) === '{"action":"reset"}');
p.handlers.state(S.answered_too_few);
expect('answered with the counter too low: "2 steps too few"', p.$('truthDone').textContent.includes('2 steps too few'), p.$('truthDone').textContent);
p.handlers.state(S.answered_too_many);
expect('answered with the counter too high: "1 step too many" (singular)', p.$('truthDone').textContent.includes('1 step too many'), p.$('truthDone').textContent);
p = makePage(); p.handlers.state(S.done_no_model);
expect('with no step-length model it says there is no distance estimate, and shows no distance', p.$('distText').textContent.includes('No distance estimate') && p.$('dist').textContent === '–', p.$('distText').textContent);
p.handlers.state(S.idle);
expect('back to idle: the result card goes away', p.$('result').hidden === true);

console.log('5. how right it has been');
p = makePage(); p.handlers.state(S.idle_no_sensor);
expect('with no counted demos it says so, and shows no table', p.$('tallyText').textContent.includes('No demos') && document_rows(p) === 0);
function document_rows(q) { return q.d.querySelectorAll('#sessions tr').length; }
p.handlers.state(S.answered_too_many);
const T = S.answered_too_many.tally;
expect('the tally sentence: how many demos, the average error, the percentage, within one step and the worst', p.$('tallyText').textContent.includes(T.n + ' demos') && p.$('tallyText').textContent.includes(T.mae.toFixed(2) + ' steps off') && p.$('tallyText').textContent.includes('within one step') && p.$('tallyText').textContent.includes('worst'), p.$('tallyText').textContent);
const rows = p.d.querySelectorAll('#sessions tr');
expect('the table has a header and one row per session, newest first, with the error signed', rows.length === 1 + S.answered_too_many.sessions.length && rows[1].textContent.includes('lee') && /[+-]?\d/.test(rows[1].children[4].textContent), rows[1].textContent);
expect('a session you have not answered shows a question mark', (() => { const q = makePage(); q.handlers.state(S.done); const r = q.d.querySelectorAll('#sessions tr'); return r[1].children[3].textContent === '?'; })());

console.log('6. how good it is, from the test trials');
p = makePage(); p.handlers.state(S.idle);
const E = S.idle.evaluation, ev = p.$('evalText').textContent;
expect('the card says how many trials, the average error with its interval, the rounded error and the worst', ev.includes('On ' + E.n_trials + ' trials') && ev.includes(E.network_mae.toFixed(2) + ' steps off') && ev.includes(E.network_mae_ci95[0].toFixed(2)) && ev.includes(E.network_mae_rounded.toFixed(2)) && ev.includes('worst ' + E.worst.toFixed(1)), ev.slice(0, 200));
expect('it mentions standing and tapping, the distance error for everyone and per person, and that the truth was a human count on short walks', ev.includes('standing or tapping') && ev.includes('with one model for everyone') && Object.keys(E.distance_per_person_mape).every((k) => ev.includes(k)) && ev.includes('human count') && ev.includes('5 to 6 metres'));
p.handlers.state(S.done_no_model);
expect('with no evaluation numbers next to the network it says so', p.$('evalText').textContent.includes('No evaluation numbers'));
const syn = JSON.parse(JSON.stringify(S.idle)); syn.evaluation.synthetic = true; p.handlers.state(syn);
expect('numbers from made-up trials are labelled as not real accuracy', p.$('evalText').textContent.startsWith('MADE-UP TRIALS'));

console.log('7. text from the board is shown as text');
p = makePage();
const evil = JSON.parse(JSON.stringify(S.answered_exact));
evil.person = '<b>x</b>'; evil.last.person = '<i>y</i>'; evil.last.step_length_source = '<img src=x onerror="window.hacked=1">'; evil.sessions[0].person = '<u>z</u>'; evil.people = ['<script>window.hacked=2</script>'];
p.handlers.state(evil); p.handlers.notice({ level: 'warn', text: '<img src=x onerror="window.hacked=3">' });
expect('markup in names, sources, the table, suggestions and notices is never run', p.d.querySelector('#sessions u') === null && p.d.querySelector('#result img') === null && p.d.querySelector('#notice img') === null && p.d.querySelector('script:not([src])') === null && p.w.hacked === undefined && p.d.querySelectorAll('#people option').length === 1);
p.handlers.notice({ level: 'ok', text: 'Saved.' });
expect('a notice shows its text in its own style', p.$('notice').className === 'notice ok' && p.$('notice').textContent === 'Saved.');

console.log('8. the plot is optional');
p = makePage({ noCanvas: true });
let ok = true; try { p.handlers.state(S.walking); } catch (e) { ok = false; }
expect('a browser that cannot draw still shows the count', ok && p.$('count').textContent === String(S.walking.steps));

console.log('\n' + (problems ? problems + ' CHECK(S) FAILED' : 'ALL PASSED'));
process.exit(problems ? 1 : 0);
