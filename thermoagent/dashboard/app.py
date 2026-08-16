"""Small standard-library HTTP application for live/replay dashboard use."""

from __future__ import annotations

import argparse
import json
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Sequence, Union
from urllib.parse import parse_qs, urlparse

from ..human_environment import HumanScenarioConfig
from ..human_runner import HumanOperatorEpisodeRunner, write_human_episode
from .replay import DashboardReplay, frame_svg
from .v4 import V4DashboardReplay, frame_svg_v4
from .v5 import V5DashboardReplay, frame_svg_v5
from .v6 import V6DashboardReplay, frame_svg_v6
from ..v6_experiments import read_episode_json


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ThermoHITL Operator Dashboard</title>
<style>
:root{--ink:#18212f;--muted:#657285;--line:#cad2dd;--bg:#eef2f6;--panel:#fff;--blue:#4c78a8;--red:#e45756;--teal:#72b7b2;--gold:#f2cf5b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,-apple-system,Segoe UI,sans-serif}.bar{display:flex;align-items:center;gap:16px;padding:14px 22px;background:#172334;color:white}.bar h1{font-size:18px;margin:0}.bar small{color:#c8d2df}.controls{margin-left:auto;display:flex;gap:8px;align-items:center}button,select,input{font:inherit}.controls button{border:1px solid #657285;background:#26384a;color:white;border-radius:5px;padding:6px 10px;cursor:pointer}.layout{display:grid;grid-template-columns:minmax(560px,1.35fr) minmax(360px,.8fr);gap:14px;padding:14px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:9px;box-shadow:0 1px 2px #0000000a;padding:14px}.panel h2{font-size:14px;margin:0 0 10px}.network{min-height:440px}.right{display:grid;gap:14px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.metric{background:#f6f8fa;border:1px solid #e0e5eb;border-radius:6px;padding:9px}.metric b{display:block;font-size:18px;margin-top:3px}.muted{color:var(--muted);font-size:12px}.phase{height:190px}.queue{max-height:190px;overflow:auto}.queue .item{padding:7px;border-bottom:1px solid #edf0f4}.footer{padding:0 18px 16px;color:var(--muted);font-size:12px}.badge{display:inline-block;border-radius:12px;padding:2px 8px;background:#e8eef5;color:#294766}.warning{background:#fde8e7;color:#922}.split{display:grid;grid-template-columns:1fr 1fr;gap:14px}svg text{font-family:system-ui,-apple-system,Segoe UI,sans-serif}.interventions{font-family:ui-monospace,SFMono-Regular,monospace;font-size:11px;max-height:140px;overflow:auto;white-space:pre-wrap}
</style></head><body>
<div class="bar"><h1>ThermoHITL</h1><small id="meta">loading replay…</small><div class="controls"><button id="rewind">↤</button><button id="play">Play</button><button id="step">Step</button><button id="alert">Jump to alert</button><label>Step <input id="slider" type="range" min="0" value="0"></label><button id="analysis">Evaluator analysis</button><button id="export">Export SVG</button></div></div>
<main class="layout"><section class="panel network"><h2>Network and autonomy</h2><svg id="network" width="100%" viewBox="0 0 700 430"></svg></section><section class="right"><div class="panel"><h2>Thermodynamic system view</h2><div class="metrics" id="metrics"></div></div><div class="panel phase"><h2>Energy–entropy phase plane</h2><svg id="phase" width="100%" height="150" viewBox="0 0 400 150"></svg></div><div class="panel"><h2>Operator workload</h2><div class="metrics" id="workload"></div></div></section><section class="panel"><h2>Alert queue</h2><div id="queue" class="queue"></div></section><section class="panel"><h2>Explanation and bounded intervention</h2><div id="explanation"></div><div id="interventions" class="interventions"></div></section></main>
<section id="evaluator-panel" class="panel" hidden style="margin:0 14px 14px;border:2px solid var(--red)"><h2 style="color:#922">Evaluator-only matched counterfactual replay</h2><p class="warning">Privileged analysis: these outcomes were unavailable to agents, the delegation controller, and the simulated operator.</p><div id="evaluator" class="interventions"></div></section>
<div class="footer">The dashboard renders the same hashed payload consumed by the simulated operator. It is technical preparation for a future approved human study, not human-subject evidence.</div>
<script>
let meta,step=0,timer=null,frame=null,analysisOpen=false;const $=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function loadMeta(){meta=await(await fetch('/api/metadata')).json();$('meta').textContent=`${meta.run_id} · ${meta.application} · ${meta.method} · simulated operator`;$('slider').max=meta.steps-1;await load(0)}
async function load(s){step=Math.max(0,Math.min(meta.steps-1,Number(s)));$('slider').value=step;frame=await(await fetch(`/api/frame?step=${step}`)).json();render();if(analysisOpen)await loadEvaluator()}
async function loadEvaluator(){const response=await fetch(`/api/evaluator-frame?step=${step}`);if(!response.ok){$('evaluator').textContent='Evaluator replay is unavailable for this study version.';return}const value=await response.json();$('evaluator').textContent=JSON.stringify(value,null,2)}
function render(){renderNetwork();const t=frame.thermodynamics;const names=['energy','effective_temperature','free_energy','entropy','tsallis_q_0_5','gini_simpson','pooled_uncertainty','entropy_slope','disagreement','graph_disagreement','consensus_confidence','consensus_residual','service_loss','autonomy_level'];$('metrics').innerHTML=names.map(k=>`<div class="metric"><span class="muted">${esc(k.replaceAll('_',' '))}</span><b>${typeof t[k]==='number'?t[k].toFixed(3):esc(t[k])}</b></div>`).join('');renderPhase();const w=frame.workload;$('workload').innerHTML=Object.entries(w).map(([k,v])=>`<div class="metric"><span class="muted">${esc(k.replaceAll('_',' '))}</span><b>${typeof v==='number'?v.toFixed(2):esc(v)}</b></div>`).join('');$('queue').innerHTML=frame.alert_queue.length?frame.alert_queue.map(q=>`<div class="item"><span class="badge">${esc(q.incident_id)}</span> ${esc(q.proposed_action??'no proposal')} · JS ${typeof q.disagreement==='number'?q.disagreement.toFixed(3):'n/a'} · consensus ${typeof q.consensus==='number'?q.consensus.toFixed(3):'n/a'}</div>`).join(''):'<span class="muted">No queued alerts</span>';const x=frame.explanation;$('explanation').innerHTML=`<p><span class="badge">${esc(x.view_condition)}</span> Alert driver: <b>${esc(x.alert_reason??'none')}</b></p><p class="muted">${esc(JSON.stringify(x.prediction))}</p><p class="muted">Alternatives: ${esc((frame.alternatives||[]).join(' · '))}</p><p class="muted">Payload hashes: ${esc(frame.view_hashes.join(', ')||'none')}</p>`;$('interventions').textContent=JSON.stringify(frame.interventions.slice(-4),null,2)}
function pos(n,i,N){if(Array.isArray(n.location))return [350+250*n.location[0],215+170*n.location[1]];let a=2*Math.PI*i/N;return [350+250*Math.cos(a),215+170*Math.sin(a)]}
function edge(h,e,P,color,width,dash=''){if(P[e[0]]&&P[e[1]])return h+`<line x1="${P[e[0]][0]}" y1="${P[e[0]][1]}" x2="${P[e[1]][0]}" y2="${P[e[1]][1]}" stroke="${color}" stroke-width="${width}" ${dash?`stroke-dasharray="${dash}"`:''}/>`;return h}
function renderNetwork(){const svg=$('network'),nodes=frame.network.nodes||[],P={};nodes.forEach((n,i)=>P[n.agent_id]=pos(n,i,nodes.length));let h='';for(const e of frame.network.service_edges||[])h=edge(h,e,P,'#009e73',3);for(const e of frame.network.logistics_edges||frame.network.physical_edges||[])h=edge(h,e,P,'#7a8798',2.3);for(const e of frame.network.communication_edges||[])h=edge(h,e,P,'#0072b2',1.3,'5 4');for(const e of frame.network.authorized_emergency_edges||[])h=edge(h,e,P,'#d55e00',5);const C={low:'#72b7b2',nominal:'#f2cf5b',high:'#e45756'};nodes.forEach(n=>{let [x,y]=P[n.agent_id],r=17+2*(n.autonomy_level||0);h+=`<circle cx="${x}" cy="${y}" r="${r}" fill="${C[n.energy_band]||'#b9c2cf'}" stroke="#26384a" stroke-width="2"/><text x="${x}" y="${y+r+15}" text-anchor="middle" font-size="11">${esc(n.agent_id)}</text><text x="${x}" y="${y+4}" text-anchor="middle" font-size="10">L${n.autonomy_level||0}</text>`});svg.innerHTML=h}
function renderPhase(){const t=frame.thermodynamics,isV4=t.standardized_energy!==undefined;let xv=isV4?t.entropy_anomaly:t.entropy,yv=isV4?t.standardized_energy:t.energy;if(xv===null||yv===null){$('phase').innerHTML='<text x="200" y="80" text-anchor="middle" font-size="12">No operator-authorized thermodynamic payload yet</text>';return}const x=isV4?45+300*Math.max(0,Math.min(1,xv/10)):45+300*Math.max(0,Math.min(1,xv));const y=isV4?125-100*Math.max(0,Math.min(1,(yv+2)/10)):125-100*Math.max(0,Math.min(1,yv));const score=t.intervention_score,active=score!==null&&score>=t.prospective_threshold;$('phase').innerHTML=`<rect x="45" y="25" width="300" height="100" fill="#eef3f7"/><line x1="45" y1="125" x2="345" y2="125" stroke="#26384a"/><line x1="45" y1="125" x2="45" y2="25" stroke="#26384a"/><line x1="195" y1="125" x2="195" y2="25" stroke="#d55e00" stroke-dasharray="6 4"/><circle cx="${x}" cy="${y}" r="7" fill="${active?'#d55e00':'#0072b2'}"/><text x="195" y="145" text-anchor="middle" font-size="11">${isV4?'entropy anomaly (0–10)':'operational entropy'}</text><text x="8" y="80" font-size="11" transform="rotate(-90 8 80)">${isV4?'standardized energy (−2–8)':'operational energy'}</text><text x="335" y="39" text-anchor="end" font-size="10">score ${score===null?'n/a':score.toFixed(2)} / τ ${t.prospective_threshold??'n/a'}</text>`}
$('slider').oninput=e=>load(e.target.value);$('step').onclick=()=>load(step+1);$('rewind').onclick=()=>load(0);$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='Play'}else{timer=setInterval(()=>{if(step>=meta.steps-1){clearInterval(timer);timer=null;$('play').textContent='Play'}else load(step+1)},650);$('play').textContent='Pause'}};$('alert').onclick=async()=>{for(let s=step+1;s<meta.steps;s++){let f=await(await fetch(`/api/frame?step=${s}`)).json();if(f.view_hashes.length){return load(s)}}};$('analysis').onclick=async()=>{analysisOpen=!analysisOpen;$('evaluator-panel').hidden=!analysisOpen;$('analysis').textContent=analysisOpen?'Hide evaluator analysis':'Evaluator analysis';if(analysisOpen)await loadEvaluator()};$('export').onclick=()=>window.open(`/export/state.svg?step=${step}`,'_blank');loadMeta();
</script></body></html>"""


def _live_replay() -> DashboardReplay:
    directory = Path(tempfile.mkdtemp(prefix="thermohitl-dashboard-"))
    config = HumanScenarioConfig(
        application="commercial",
        seed=12001,
        horizon=20,
        n_agents=8,
        topology="human_v3_development",
        disruption="moderate",
        decision_interval=2,
        communication_budget=100,
        operator_seed=22001,
    )
    runner = HumanOperatorEpisodeRunner(
        config, "periodic_human_review", enable_counterfactual_probes=False
    )
    result = runner.run("dashboard-live-sample")
    write_human_episode(result, runner.env.ledger, directory)
    return DashboardReplay(directory / "episode.json")


def _load_replay(episode_path: Path) -> Union[DashboardReplay, V4DashboardReplay, V5DashboardReplay, V6DashboardReplay]:
    payload = read_episode_json(episode_path) if episode_path.name.endswith(".json.gz") else json.loads(episode_path.read_text(encoding="utf-8"))
    if payload.get("study") == "Generalized Entropic Consensus V6":
        return V6DashboardReplay(episode_path)
    if payload.get("study") == "ThermoHITL v5":
        return V5DashboardReplay(episode_path)
    if "information_condition" in payload and "regime" in payload:
        return V4DashboardReplay(episode_path)
    return DashboardReplay(episode_path)


def serve(replay: Union[DashboardReplay, V4DashboardReplay, V5DashboardReplay, V6DashboardReplay], host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib callback
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/metadata":
                self._send(json.dumps(replay.metadata(), allow_nan=False).encode("utf-8"), "application/json")
            elif parsed.path == "/api/frame":
                step = int(query.get("step", ["0"])[0])
                self._send(json.dumps(replay.frame(step).as_dict(), allow_nan=False).encode("utf-8"), "application/json")
            elif parsed.path == "/api/evaluator-frame":
                if not isinstance(replay, V6DashboardReplay):
                    self._send(b"evaluator replay unavailable", "text/plain", 404)
                    return
                step = int(query.get("step", ["0"])[0])
                self._send(
                    json.dumps(replay.evaluator_frame(step), allow_nan=False).encode("utf-8"),
                    "application/json",
                )
            elif parsed.path == "/export/state.svg":
                step = int(query.get("step", ["0"])[0])
                frame = replay.frame(step)
                rendered = (
                    frame_svg_v6(frame) if isinstance(replay, V6DashboardReplay)
                    else frame_svg_v5(frame) if isinstance(replay, V5DashboardReplay)
                    else frame_svg_v4(frame) if isinstance(replay, V4DashboardReplay)
                    else frame_svg(frame)
                )
                self._send(rendered.encode("utf-8"), "image/svg+xml")
            elif parsed.path == "/export/state.json":
                step = int(query.get("step", ["0"])[0])
                self._send(json.dumps(replay.frame(step).as_dict(), indent=2, allow_nan=False).encode("utf-8"), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, int(port)), Handler)
    print("ThermoHITL dashboard: http://%s:%d" % (host, port), flush=True)
    server.serve_forever()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ThermoHITL operator dashboard")
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if bool(args.episode) == bool(args.live):
        parser.error("choose exactly one of --episode or --live")
    replay = _live_replay() if args.live else _load_replay(args.episode)
    serve(replay, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
