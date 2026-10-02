"""Single-file dashboard served at ``/`` (no build step, no external assets)."""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FlexTrain</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --fg:#1d2330; --muted:#667085; --line:#e4e7ec; --accent:#2563eb;
          --ok:#12805c; --warn:#b54708; --err:#b42318; --run:#2563eb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f1218; --card:#171b24; --fg:#e6e8ee; --muted:#98a2b3; --line:#2a3040; --accent:#6ea0ff;
            --ok:#3ccb8f; --warn:#f5a524; --err:#f97066; --run:#6ea0ff; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; background:var(--bg); color:var(--fg); }
  header { display:flex; align-items:baseline; gap:12px; padding:20px 24px 8px; }
  header h1 { font-size:20px; margin:0; }
  header span { color:var(--muted); }
  main { padding:8px 24px 32px; max-width:1200px; }
  .tiles { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; }
  .tile { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:120px; }
  .tile b { display:block; font-size:22px; }
  .tile small { color:var(--muted); text-transform:uppercase; letter-spacing:.04em; font-size:11px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:auto; }
  table { width:100%; border-collapse:collapse; }
  th, td { text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { color:var(--muted); font-weight:600; font-size:12px; }
  tbody tr { cursor:pointer; }
  tbody tr:hover, tbody tr.sel { background:color-mix(in srgb, var(--accent) 8%, transparent); }
  td.num { font-variant-numeric: tabular-nums; }
  .pill { padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; border:1px solid currentColor; }
  .running { color:var(--run); } .completed { color:var(--ok); } .preempted { color:var(--warn); }
  .failed, .stale { color:var(--err); }
  #chart { margin-top:16px; padding:16px; }
  #chart h2 { font-size:15px; margin:0 0 8px; }
  svg text { fill:var(--muted); font-size:11px; }
  .empty { padding:24px; color:var(--muted); }
</style>
</head>
<body>
<header><h1>FlexTrain</h1><span id="updated">loading…</span></header>
<main>
  <div class="tiles" id="tiles"></div>
  <div class="card">
    <table>
      <thead><tr><th>Run</th><th>Status</th><th>Step</th><th>Epoch</th><th>Loss</th><th>Samples/s</th>
        <th>World</th><th>Restarts</th><th>Last checkpoint</th><th>Updated</th></tr></thead>
      <tbody id="rows"><tr><td colspan="10" class="empty">No runs yet. Start one with <code>flextrain launch</code>.</td></tr></tbody>
    </table>
  </div>
  <div class="card" id="chart" hidden><h2 id="chart-title"></h2><svg id="svg" width="100%" height="240"></svg></div>
</main>
<script>
let selected = null;
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (v, d=4) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : (typeof v === "number" ? +v.toFixed(d) : esc(v));
const ago = s => s < 60 ? `${Math.round(s)}s ago` : s < 3600 ? `${Math.round(s/60)}m ago` : `${Math.round(s/3600)}h ago`;

async function refresh() {
  try {
    const res = await fetch("/api/jobs"); const data = await res.json();
    const counts = data.counts || {};
    document.getElementById("tiles").innerHTML = ["running","completed","preempted","failed"].map(k =>
      `<div class="tile"><small>${k}</small><b class="${k}">${counts[k] || 0}</b></div>`).join("");
    const rows = data.jobs.map(j => {
      const status = j.stale ? "stale" : j.status;
      const ckpt = j.last_checkpoint_step != null ? `step ${j.last_checkpoint_step}` : "–";
      return `<tr data-id="${esc(j.run_id)}" class="${j.run_id === selected ? "sel" : ""}">
        <td>${esc(j.run_id)}</td><td><span class="pill ${status}">${esc(status)}</span></td>
        <td class="num">${fmt(j.step, 0)}${j.max_steps ? " / " + j.max_steps : ""}</td><td class="num">${fmt(j.epoch, 0)}</td>
        <td class="num">${fmt(j.loss)}</td><td class="num">${fmt(j.samples_per_s, 1)}</td>
        <td class="num">${fmt(j.world_size, 0)}</td><td class="num">${fmt(j.restart_count, 0)}</td>
        <td>${ckpt}</td><td>${ago(j.seconds_since_update)}</td></tr>`;
    }).join("");
    if (rows) document.getElementById("rows").innerHTML = rows;
    document.querySelectorAll("#rows tr[data-id]").forEach(tr => tr.onclick = () => { selected = tr.dataset.id; refresh(); });
    document.getElementById("updated").textContent = "updated " + new Date().toLocaleTimeString();
    if (selected) drawChart(selected);
  } catch (e) { document.getElementById("updated").textContent = "API unreachable"; }
}

async function drawChart(runId) {
  const res = await fetch(`/api/jobs/${encodeURIComponent(runId)}/metrics?tail=5000`);
  if (!res.ok) return;
  const pts = (await res.json()).metrics.filter(m => typeof m.loss === "number").map(m => [m.step, m.loss]);
  const box = document.getElementById("chart"); box.hidden = false;
  document.getElementById("chart-title").textContent = `Training loss — ${runId}`;
  const svg = document.getElementById("svg"); const W = svg.clientWidth || 800, H = 240, P = 40;
  if (pts.length < 2) { svg.innerHTML = `<text x="${P}" y="${H/2}">not enough points yet</text>`; return; }
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const sx = x => P + (x - x0) / Math.max(1e-9, x1 - x0) * (W - 2 * P);
  const sy = y => H - P + (P - H + P) * (y - y0) / Math.max(1e-9, y1 - y0);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
  svg.innerHTML = `<line x1="${P}" y1="${H-P}" x2="${W-P}" y2="${H-P}" stroke="var(--line)"/>
    <path d="${d}" fill="none" stroke="var(--accent)" stroke-width="2"/>
    <text x="${P}" y="${H-P+16}">step ${x0}</text><text x="${W-P}" y="${H-P+16}" text-anchor="end">step ${x1}</text>
    <text x="4" y="${sy(y1)+4}">${y1.toFixed(3)}</text><text x="4" y="${sy(y0)+4}">${y0.toFixed(3)}</text>`;
}
refresh(); setInterval(refresh, 5000);
</script>
</body>
</html>
"""
