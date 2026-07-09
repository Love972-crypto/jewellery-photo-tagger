from __future__ import annotations

import mimetypes
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


@dataclass(frozen=True)
class DownloadServerInfo:
    base_url: str
    runtime_root: Path


def start_download_server(project_root: Path, preferred_port: int = 8765) -> DownloadServerInfo:
    runtime_root = (project_root / ".runtime").resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)

    handler = _build_handler(runtime_root)
    last_error: OSError | None = None
    for port in range(preferred_port, preferred_port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            return DownloadServerInfo(base_url=f"http://127.0.0.1:{port}", runtime_root=runtime_root)
        except OSError as exc:
            last_error = exc

    raise RuntimeError(f"Could not start local download server: {last_error}")


def download_url(info: DownloadServerInfo, path: Path) -> str:
    relative = path.resolve().relative_to(info.runtime_root).as_posix()
    return f"{info.base_url}/download?file={quote(relative)}"


def _build_handler(runtime_root: Path):
    class DownloadHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/download":
                self.send_error(404)
                return

            file_values = parse_qs(parsed.query).get("file", [])
            if not file_values:
                self.send_error(400, "Missing file")
                return

            try:
                relative = unquote(file_values[0]).replace("\\", "/").lstrip("/")
                target = (runtime_root / relative).resolve()
                target.relative_to(runtime_root)
                if not target.is_file():
                    self.send_error(404)
                    return
            except Exception:
                self.send_error(400, "Bad file")
                return

            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(target.stat().st_size))
            self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
            self.end_headers()

            with target.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    return DownloadHandler
