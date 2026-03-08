from __future__ import annotations

import json
import pathlib
import threading
import time
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from pointer_targets import (
    AmbiguousTargetError,
    TargetResolutionError,
    TargetResolver,
    TargetSpec,
)


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pointer Control</title>
  <style>
    :root {
      --bg: #07151b;
      --panel: #10252d;
      --accent: #ffd166;
      --text: #ecf6f8;
      --muted: #9bb8bf;
      --danger: #ef476f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, sans-serif;
      background:
        radial-gradient(circle at top, rgba(255, 209, 102, 0.15), transparent 35%),
        linear-gradient(180deg, #07151b 0%, #091e26 100%);
      color: var(--text);
    }
    main { max-width: 980px; margin: 0 auto; padding: 24px; display: grid; gap: 18px; }
    section {
      background: rgba(16, 37, 45, 0.88);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
    }
    h1, h2 { margin: 0 0 12px; }
    h1 { font-size: 1.9rem; }
    h2 { font-size: 1.1rem; color: var(--accent); }
    .row { display: flex; flex-wrap: wrap; gap: 10px; }
    input, select, button {
      border-radius: 10px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 10px 12px;
      font: inherit;
      color: var(--text);
      background: rgba(0, 0, 0, 0.15);
    }
    input, select { min-width: 180px; flex: 1; }
    button {
      background: var(--accent);
      color: #1d1b17;
      border: none;
      cursor: pointer;
      font-weight: 700;
    }
    button.secondary { background: rgba(255,255,255,0.08); color: var(--text); }
    button.danger { background: var(--danger); color: white; }
    ul { list-style: none; padding: 0; margin: 0; display: grid; gap: 10px; }
    li {
      padding: 12px;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.05);
      display: grid;
      gap: 8px;
    }
    .muted { color: var(--muted); }
    pre {
      margin: 0;
      white-space: pre-wrap;
      font-size: 0.92rem;
      color: var(--muted);
    }
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Pointer Control</h1>
      <div class="row">
        <input id="query" placeholder="Target name, ID, or catalog entry">
        <select id="kind">
          <option value="auto">Auto</option>
          <option value="satellite">Earth satellite</option>
          <option value="solar-system">Solar system</option>
          <option value="spacecraft">Interplanetary spacecraft</option>
          <option value="star">Star</option>
          <option value="constellation">Constellation</option>
          <option value="dso">Deep sky</option>
        </select>
        <button onclick="searchTargets()">Search</button>
      </div>
    </section>

    <section>
      <h2>Active Status</h2>
      <pre id="status">Loading…</pre>
    </section>

    <section>
      <h2>Search Results</h2>
      <ul id="results"><li class="muted">No search yet.</li></ul>
    </section>

    <section>
      <h2>Saved Presets</h2>
      <div class="row">
        <input id="preset-name" placeholder="Preset name">
        <button class="secondary" onclick="savePreset()">Save current target</button>
      </div>
      <ul id="presets"><li class="muted">Loading presets…</li></ul>
    </section>
  </main>
  <script>
    async function request(path, options) {
      const response = await fetch(path, options);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || response.statusText);
      }
      return payload;
    }

    function renderTargetRow(result, onClickLabel, onClick) {
      const li = document.createElement('li');
      const title = document.createElement('div');
      title.textContent = `${result.display_name} [${result.kind}]`;
      li.appendChild(title);
      if (result.description) {
        const meta = document.createElement('div');
        meta.className = 'muted';
        meta.textContent = result.description;
        li.appendChild(meta);
      }
      const button = document.createElement('button');
      button.textContent = onClickLabel;
      button.onclick = onClick;
      li.appendChild(button);
      return li;
    }

    async function refreshStatus() {
      try {
        const payload = await request('/api/status');
        document.getElementById('status').textContent = JSON.stringify(payload, null, 2);
      } catch (error) {
        document.getElementById('status').textContent = error.message;
      }
    }

    async function refreshPresets() {
      try {
        const payload = await request('/api/presets');
        const list = document.getElementById('presets');
        list.innerHTML = '';
        if (!payload.presets.length) {
          list.innerHTML = '<li class="muted">No presets saved.</li>';
          return;
        }
        for (const preset of payload.presets) {
          const li = document.createElement('li');
          li.innerHTML = `<strong>${preset.name}</strong><div class="muted">${preset.spec.kind}: ${preset.spec.query}</div>`;
          const row = document.createElement('div');
          row.className = 'row';

          const activate = document.createElement('button');
          activate.textContent = 'Activate';
          activate.onclick = async () => {
            await request(`/api/presets/${encodeURIComponent(preset.name)}/activate`, { method: 'POST' });
            await refreshStatus();
          };
          row.appendChild(activate);

          const remove = document.createElement('button');
          remove.className = 'danger';
          remove.textContent = 'Delete';
          remove.onclick = async () => {
            await request(`/api/presets/${encodeURIComponent(preset.name)}`, { method: 'DELETE' });
            await refreshPresets();
          };
          row.appendChild(remove);
          li.appendChild(row);
          list.appendChild(li);
        }
      } catch (error) {
        document.getElementById('presets').innerHTML = `<li class="muted">${error.message}</li>`;
      }
    }

    async function searchTargets() {
      const query = document.getElementById('query').value.trim();
      const kind = document.getElementById('kind').value;
      if (!query) return;
      const payload = await request(`/api/search?q=${encodeURIComponent(query)}&kind=${encodeURIComponent(kind)}`);
      const list = document.getElementById('results');
      list.innerHTML = '';
      if (!payload.matches.length) {
        list.innerHTML = '<li class="muted">No matches.</li>';
        return;
      }
      for (const result of payload.matches) {
        list.appendChild(renderTargetRow(result, 'Track', async () => {
          await request('/api/active-target', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ spec: {
              kind: result.kind,
              query: result.query,
              identifier: result.identifier,
              source: result.source,
            }})
          });
          await refreshStatus();
        }));
      }
    }

    async function savePreset() {
      const name = document.getElementById('preset-name').value.trim();
      if (!name) return;
      await request('/api/presets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
      });
      document.getElementById('preset-name').value = '';
      await refreshPresets();
    }

    refreshStatus();
    refreshPresets();
    setInterval(refreshStatus, 1500);
  </script>
</body>
</html>
"""


class PointerTrackingService:
    def __init__(
        self,
        *,
        resolver: TargetResolver,
        observer_cfg: dict,
        pointer_factory,
        pointer_config: dict,
        preset_store: pathlib.Path,
        disable_servo: bool = False,
        update_seconds: float = 1.0,
    ):
        self.resolver = resolver
        self.observer_cfg = observer_cfg
        self.pointer_factory = pointer_factory
        self.pointer_config = pointer_config
        self.disable_servo = disable_servo
        self.update_seconds = update_seconds
        self.preset_store = preset_store
        self.preset_store.parent.mkdir(parents=True, exist_ok=True)
        self._observer_context = self.resolver.build_observer_context(observer_cfg)
        self._pointer = None
        self._active_spec: TargetSpec | None = None
        self._active_target = None
        self._last_state = None
        self._last_command = None
        self._last_error = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)

        if not self.disable_servo:
            if (
                self.pointer_config["servo"].get("base_reference_ticks") is None
                or self.pointer_config["servo"].get("alt_reference_ticks") is None
            ):
                raise RuntimeError("Missing servo reference ticks; run --set-reference before enabling web tracking.")
            self._pointer = self.pointer_factory(self.pointer_config)
            self._pointer.open()

    def start(self):
        self._thread.start()

    def close(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._pointer is not None:
            self._pointer.close()

    def status(self) -> dict:
        with self._lock:
            return {
                "active_target": asdict(self._active_spec) if self._active_spec is not None else None,
                "state": self._last_state.as_dict() if self._last_state is not None else None,
                "servo_command": self._last_command,
                "error": self._last_error,
                "servo_enabled": self._pointer is not None,
            }

    def search(self, query: str, kind: str) -> list[dict]:
        matches = self.resolver.search(query, kind=kind)
        return [asdict(match) for match in matches]

    def set_active_target(self, spec_payload: dict):
        spec = TargetSpec(
            kind=spec_payload.get("kind", "auto"),
            query=spec_payload.get("query", ""),
            source=spec_payload.get("source"),
            identifier=spec_payload.get("identifier"),
        )
        target = self.resolver.resolve(spec, self.observer_cfg)
        with self._lock:
            self._active_spec = spec
            self._active_target = target
            self._last_error = None

    def list_presets(self) -> list[dict]:
        if not self.preset_store.exists():
            return []
        payload = json.loads(self.preset_store.read_text())
        return payload.get("presets", [])

    def save_current_preset(self, name: str):
        with self._lock:
            if self._active_spec is None:
                raise RuntimeError("No active target to save.")
            presets = self.list_presets()
            filtered = [preset for preset in presets if preset["name"] != name]
            filtered.append({"name": name, "spec": asdict(self._active_spec)})
            self.preset_store.write_text(json.dumps({"presets": filtered}, indent=2, sort_keys=True) + "\n")

    def delete_preset(self, name: str):
        presets = [preset for preset in self.list_presets() if preset["name"] != name]
        self.preset_store.write_text(json.dumps({"presets": presets}, indent=2, sort_keys=True) + "\n")

    def activate_preset(self, name: str):
        for preset in self.list_presets():
            if preset["name"] == name:
                self.set_active_target(preset["spec"])
                return
        raise RuntimeError(f"Preset '{name}' was not found.")

    def _run_loop(self):
        while not self._stop_event.wait(self.update_seconds):
            with self._lock:
                active_spec = self._active_spec
                active_target = self._active_target
            if active_spec is None or active_target is None:
                continue

            target_time = self.resolver.ts.now()
            try:
                try:
                    state = active_target.state_at(target_time, self._observer_context)
                except TargetResolutionError:
                    active_target = self.resolver.resolve(active_spec, self.observer_cfg)
                    state = active_target.state_at(target_time, self._observer_context)
                command = None
                if self._pointer is not None:
                    command = self._pointer.point(state.az_deg, state.alt_deg)
                with self._lock:
                    self._active_target = active_target
                    self._last_state = state
                    self._last_command = command
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)


def serve_web_app(
    *,
    resolver: TargetResolver,
    observer_cfg: dict,
    pointer_factory,
    pointer_config: dict,
    disable_servo: bool,
    host: str,
    port: int,
    update_seconds: float,
    preset_store: pathlib.Path,
):
    service = PointerTrackingService(
        resolver=resolver,
        observer_cfg=observer_cfg,
        pointer_factory=pointer_factory,
        pointer_config=pointer_config,
        disable_servo=disable_servo,
        update_seconds=update_seconds,
        preset_store=preset_store,
    )
    service.start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send_json(self, payload: dict, *, status: int = HTTPStatus.OK):
            data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/status":
                self._send_json(service.status())
                return

            if parsed.path == "/api/search":
                params = parse_qs(parsed.query)
                query = params.get("q", [""])[0]
                kind = params.get("kind", ["auto"])[0]
                self._send_json({"matches": service.search(query, kind)})
                return

            if parsed.path == "/api/presets":
                self._send_json({"presets": service.list_presets()})
                return

            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/active-target":
                    payload = self._read_json()
                    service.set_active_target(payload["spec"])
                    self._send_json({"ok": True})
                    return

                if parsed.path == "/api/presets":
                    payload = self._read_json()
                    service.save_current_preset(payload["name"])
                    self._send_json({"ok": True})
                    return

                if parsed.path.startswith("/api/presets/") and parsed.path.endswith("/activate"):
                    name = parsed.path[len("/api/presets/"):-len("/activate")].strip("/")
                    service.activate_preset(name)
                    self._send_json({"ok": True})
                    return
            except AmbiguousTargetError as exc:
                self._send_json(
                    {"error": str(exc), "matches": [asdict(match) for match in exc.matches]},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def do_DELETE(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/presets/"):
                name = parsed.path[len("/api/presets/"):].strip("/")
                try:
                    service.delete_preset(name)
                    self._send_json({"ok": True})
                    return
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return

            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        print(f"Serving pointer control UI at http://{host}:{port}")
        server.serve_forever()
    finally:
        server.server_close()
        service.close()
