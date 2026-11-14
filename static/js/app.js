/* PhishGuard frontend */

const SAMPLES = {
  phish1: `Subject: Urgent: Your account will be suspended within 24 hours

Dear Valued Customer,

We detected unusual sign-in activity on your account from a new device.
To protect your information you must verify your password immediately.
Failure to act within 24 hours will result in permanent account suspension.

Click here to secure your account now:
https://secure-login-verify.xyz/account/update?token=9f3a2b

IT Security Team`,

  phish2: `Subject: Invoice #INV-98421 — Wire transfer required today

Hello,

Please process the attached invoice (invoice_payment.exe) before close of business.
Funds will be released only after the bank transfer is confirmed.

Open the attachment and confirm your identity:
https://paypa1-secure.top/wire/transfer

Accounts Payable`,

  phish3: `Subject: CONGRATULATIONS! You are a finalist — claim your prize

You have been selected as a finalist in the Travel Giveaway!
Complete the registration form and include your Confirmation Number.
First 500 finalists receive free entry into a cash drawing.

CLICK HERE to claim your prize:
https://travel-prize-claim.info/register?id=77621`,

  phish4: `Subject: Security Alert: Unusual transaction blocked

Dear Customer,

We temporarily blocked a transaction of $1,850 from your account.
Confirm the payment within 12 hours or the hold becomes permanent.

Verify your identity here:
https://bank-secure-auth.top/confirm

Customer Security Department`,

  phish5: `Subject: Action required: Payroll direct-deposit update

HR Notice: Your direct-deposit details must be re-verified this week
or your next salary payment may be delayed.

Sign in to the payroll portal and confirm your bank account:
https://payroll-hr-update.club/verify

Human Resources`,

  obfusc: `Subject: Action required — confirm your Microsoft identity

Your micros0ft account needs immediate verification.
Sign in at the official portal to avoid suspension:

https://login-micros0ft-secure.club/auth/verify

An OTP will be sent after you enter your credentials.
Microsoft Account Protection`,

  legit1: `Subject: Sprint standup notes — 3 Nov

Team,

Attached are the notes from this morning's standup.
Agenda items: milestone review, project status, next sprint planning.

See you at the afternoon meeting in Conference Room B.

Best,
Alex`,

  legit2: `Subject: RE: Project timeline update

Hi Sarah,

Thanks for the update. I reviewed the revised timeline and it looks good.
We can proceed with the current plan for the next milestone.

Let me know if you need anything else from my side before Friday.

Regards,
Michael`,

  legit3: `Subject: Meeting invitation: Q4 planning session

Hi everyone,

Please join us for the Q4 planning session on Thursday at 10:00 AM.
Agenda and pre-read materials are attached.

Conference link will be sent the day before.

Thanks,
Operations`,

  fp1: `Subject: Monthly IT security reminder — no action required

Hi team,

This is our routine monthly reminder from Corporate IT.

- Do not share passwords
- Report suspicious messages to security@company.com
- MFA remains mandatory for all staff

There is no link to click and no password to enter.
This message is for awareness only.

Corporate Information Security
internal-only distribution`,

  fp2: `Subject: Your password will expire in 14 days

Hello,

Per company policy, your network password is scheduled to expire in 14 days.

When you are ready, change it on your work PC using Ctrl+Alt+Delete → Change a password.
You do not need to visit any website or enter credentials in email.

If you already changed it this month, you can ignore this notice.

IT Service Desk
service-desk@company.local`,

  fp3: `Subject: Invoice 2024-1182 from Acme Supplies (PDF attached)

Hi Accounts,

Please find attached invoice 2024-1182 for the October stationery order.
Amount due: $412.50. Payment terms: Net 30 as agreed in our PO-8891.

No portal login is required — reply to this thread if the PO number looks wrong.

Regards,
Acme Supplies Billing`,

  fp4: `Subject: Your bank statement is now available

Hello,

Your monthly statement for account ending 4821 is ready in online banking.
Log in through the official mobile app or website when convenient.

This is an automated notification. No reply is needed.

Customer Communications`
};

let chart = null;
const historyPoints = []; // {t, legit, phish}

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function toast(msg, ms = 2800) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), ms);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

function renderResult(data) {
  const box = $('#classifyResult');
  const isPhish = (data.classification || '').toLowerCase() === 'phishing';
  box.className = 'result show ' + (isPhish ? 'glow-phish' : 'glow-legit');

  // Backend key is phishing_probability
  const p = data.phishing_probability
    ?? data.probability_phishing
    ?? data.prob_phishing
    ?? data.score
    ?? data.confidence
    ?? 0;
  const pct = (Number(p) * 100).toFixed(1);
  const reasons = data.reason_codes || data.reasons || [];
  const meta = data.metadata || data.features || {};

  let codesHtml = reasons.map(r => `<span class="code">${escapeHtml(String(r))}</span>`).join('');
  if (!codesHtml && data.explanation) {
    codesHtml = `<span class="code">${escapeHtml(data.explanation)}</span>`;
  }

  const metaItems = [];
  if (meta.url_count != null) metaItems.push(`<div class="meta-item"><span>URLs</span>${meta.url_count}</div>`);
  if (meta.high_risk_tld != null) metaItems.push(`<div class="meta-item"><span>High-risk TLD</span>${meta.high_risk_tld}</div>`);
  if (meta.subject_entropy != null) metaItems.push(`<div class="meta-item"><span>Subject entropy</span>${Number(meta.subject_entropy).toFixed(2)}</div>`);
  if (meta.body_entropy != null) metaItems.push(`<div class="meta-item"><span>Body entropy</span>${Number(meta.body_entropy).toFixed(2)}</div>`);
  if (meta.attachment_count != null) metaItems.push(`<div class="meta-item"><span>Attachments</span>${meta.attachment_count}</div>`);
  if (data.latency_ms != null) metaItems.push(`<div class="meta-item"><span>Latency</span>${data.latency_ms} ms</div>`);

  // fallback generic keys
  if (!metaItems.length && typeof meta === 'object') {
    Object.entries(meta).slice(0, 6).forEach(([k, v]) => {
      metaItems.push(`<div class="meta-item"><span>${escapeHtml(k)}</span>${escapeHtml(String(v))}</div>`);
    });
  }

  const modelPct = data.model_probability != null ? (Number(data.model_probability) * 100).toFixed(1) : '—';
  const lexPct = data.lexical_score != null ? (Number(data.lexical_score) * 100).toFixed(1) : '—';

  box.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;gap:0.75rem;flex-wrap:wrap;">
      <span class="pill ${isPhish ? 'phishing' : 'legitimate'}">${isPhish ? 'Phishing' : 'Legitimate'}</span>
      <span style="font-size:0.8rem;color:var(--muted)">${data.trained ? 'Multi-Vector (trained)' : 'Multi-Vector'} · ${data.latency_ms ?? '—'} ms</span>
    </div>
    <div class="prob ${isPhish ? 'phish' : 'legit'}">${pct}%</div>
    <div style="font-size:0.82rem;color:var(--muted);margin-bottom:0.35rem;">
      Hybrid phishing probability ${isPhish ? '— threat detected' : '— low risk'}
      <span style="opacity:0.75">(neural ${modelPct}% · lexical ${lexPct}%)</span>
    </div>
    ${codesHtml ? `<div class="codes">${codesHtml}</div>` : ''}
    ${metaItems.length ? `<div class="meta-grid">${metaItems.join('')}</div>` : ''}
  `;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

async function classify() {
  const text = $('#emailText').value.trim();
  if (!text) {
    toast('Paste or drop an email first');
    return;
  }
  const btn = $('#classifyBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Analyzing…';

  try {
    const data = await api('/api/classify', {
      method: 'POST',
      body: JSON.stringify({
        text,
        enable_vt: $('#enableVT').checked,
        enable_sandbox: $('#enableSandbox').checked,
      }),
    });
    renderResult(data);
    await refreshStatsAndAudit();
    toast(data.classification === 'phishing' ? '⚠ Phishing detected' : '✓ Marked legitimate');
  } catch (e) {
    toast('Error: ' + e.message);
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Classify';
  }
}

async function refreshStatsAndAudit() {
  try {
    const [stats, audit] = await Promise.all([
      api('/api/stats'),
      api('/api/audit'),
    ]);
    $('#statTotal').textContent = stats.total ?? 0;
    $('#statLegit').textContent = stats.legitimate ?? 0;
    $('#statPhish').textContent = stats.phishing ?? 0;

    // chart history
    historyPoints.push({
      t: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      legit: stats.legitimate ?? 0,
      phish: stats.phishing ?? 0,
    });
    if (historyPoints.length > 20) historyPoints.shift();
    updateChart();

    const tbody = $('#auditBody');
    if (!audit.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty">No classifications yet — try a sample</td></tr>';
      return;
    }
    tbody.innerHTML = audit.map(row => {
      const cls = (row.classification || '').toLowerCase();
      const pill = `<span class="pill ${cls === 'phishing' ? 'phishing' : 'legitimate'}">${cls || '—'}</span>`;
      const p = row.phishing_probability ?? row.probability_phishing ?? row.prob_phishing ?? row.score;
      const pStr = p != null ? Number(p).toFixed(3) : '—';
      const reasons = (row.reason_codes || row.reasons || []).slice(0, 3).join(', ') || '—';
      return `<tr>
        <td>${escapeHtml((row.ts || '').replace('T', ' ').slice(0, 19))}</td>
        <td title="${escapeHtml(row.subject || '')}">${escapeHtml((row.subject || '').slice(0, 42))}</td>
        <td>${pill}</td>
        <td>${pStr}</td>
        <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(reasons)}">${escapeHtml(reasons)}</td>
        <td>${row.latency_ms ?? '—'}</td>
      </tr>`;
    }).join('');
  } catch (e) {
    console.error('refresh failed', e);
  }
}

function updateChart() {
  const canvas = $('#statsChart');
  if (!canvas || typeof Chart === 'undefined') return;

  const labels = historyPoints.map(p => p.t);
  const legit = historyPoints.map(p => p.legit);
  const phish = historyPoints.map(p => p.phish);

  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = legit;
    chart.data.datasets[1].data = phish;
    chart.update('none');
    return;
  }

  chart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Legitimate',
          data: legit,
          borderColor: '#22c55e',
          backgroundColor: 'rgba(34,197,94,0.12)',
          tension: 0.35,
          fill: true,
          pointRadius: 3,
        },
        {
          label: 'Phishing',
          data: phish,
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239,68,68,0.12)',
          tension: 0.35,
          fill: true,
          pointRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#8b9bb4', boxWidth: 12, font: { size: 11 } },
        },
      },
      scales: {
        x: {
          ticks: { color: '#64748b', maxTicksLimit: 6, font: { size: 10 } },
          grid: { color: 'rgba(45,58,79,0.5)' },
        },
        y: {
          beginAtZero: true,
          ticks: { color: '#64748b', precision: 0, font: { size: 10 } },
          grid: { color: 'rgba(45,58,79,0.5)' },
        },
      },
    },
  });
}

function exportAudit(format) {
  api('/api/audit').then(rows => {
    if (!rows.length) {
      toast('Nothing to export yet');
      return;
    }
    let blob, name;
    if (format === 'json') {
      blob = new Blob([JSON.stringify(rows, null, 2)], { type: 'application/json' });
      name = 'phishguard-audit.json';
    } else {
      const headers = ['ts', 'subject', 'classification', 'phishing_probability', 'reason_codes', 'latency_ms'];
      const lines = [headers.join(',')];
      rows.forEach(r => {
        const p = r.phishing_probability ?? r.probability_phishing ?? r.prob_phishing ?? r.score ?? '';
        const reasons = (r.reason_codes || r.reasons || []).join('; ');
        lines.push([
          r.ts || '',
          `"${(r.subject || '').replace(/"/g, '""')}"`,
          r.classification || '',
          p,
          `"${reasons.replace(/"/g, '""')}"`,
          r.latency_ms ?? '',
        ].join(','));
      });
      blob = new Blob([lines.join('\n')], { type: 'text/csv' });
      name = 'phishguard-audit.csv';
    }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
    toast(`Exported ${rows.length} rows as ${format.toUpperCase()}`);
  }).catch(e => toast('Export failed: ' + e.message));
}

function setupDropzone() {
  const zone = $('#dropzone');
  const input = $('#fileInput');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) readFile(file);
  });
  input.addEventListener('change', () => {
    if (input.files[0]) readFile(input.files[0]);
  });
}

function readFile(file) {
  if (file.size > 2 * 1024 * 1024) {
    toast('File too large (max 2 MB)');
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    $('#emailText').value = reader.result;
    toast(`Loaded ${file.name}`);
  };
  reader.onerror = () => toast('Could not read file');
  reader.readAsText(file);
}

function setupSamples() {
  $$('.sample-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.sample;
      if (SAMPLES[key]) {
        $('#emailText').value = SAMPLES[key];
        $$('.sample-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      }
    });
  });
}

async function loadStatus() {
  try {
    const s = await api('/api/status');
    const el = $('#modelStatus');
    if (s.loaded) {
      el.textContent = s.trained ? 'Model ready · Surrogate' : 'Model loaded (untrained fallback)';
    } else {
      el.textContent = 'Model unavailable';
      el.style.background = '#3f1212';
      el.style.color = '#fecaca';
      el.style.borderColor = '#7f1d1d';
    }
  } catch {
    $('#modelStatus').textContent = 'Offline';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  $('#classifyBtn').addEventListener('click', classify);
  $('#exportCsv').addEventListener('click', () => exportAudit('csv'));
  $('#exportJson').addEventListener('click', () => exportAudit('json'));
  setupDropzone();
  setupSamples();
  loadStatus();
  refreshStatsAndAudit();
  // poll stats every 15s in case of multi-tab use
  setInterval(refreshStatsAndAudit, 15000);
});
