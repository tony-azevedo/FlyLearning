"""
html_viewer.py — Browse a folder of .html files in the browser.

Usage:
    python html_viewer.py ./Figure3/250304_F3_C1/successes_html
    python html_viewer.py  # defaults to current directory
"""

import http.server
import json
import os
import sys
import urllib.parse
import webbrowser

PORT = 8787


def build_viewer_html(html_files, folder):
    """Return the navigation page HTML."""
    files_json = json.dumps(html_files)
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>HTML Viewer — {folder}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; }}
  #toolbar {{
    display: flex; align-items: center; gap: 12px;
    padding: 8px 16px; background: #1a1a2e; color: #eee;
  }}
  #toolbar button {{
    padding: 6px 16px; font-size: 14px; cursor: pointer;
    border: 1px solid #555; border-radius: 4px;
    background: #2d2d44; color: #eee;
  }}
  #toolbar button:hover {{ background: #3d3d5c; }}
  #toolbar button:disabled {{ opacity: 0.3; cursor: default; }}
  #toolbar select {{
    padding: 4px 8px; font-size: 14px;
    background: #2d2d44; color: #eee; border: 1px solid #555; border-radius: 4px;
  }}
  #label {{ font-size: 14px; white-space: nowrap; }}
  #folder {{ font-size: 12px; color: #999; margin-left: auto; }}
  iframe {{ flex: 1; border: none; width: 100%; }}
</style>
</head><body>

<div id="toolbar">
  <button id="prev" onclick="go(-1)">&#9664; Prev</button>
  <button id="next" onclick="go(1)">Next &#9654;</button>
  <select id="dropdown" onchange="jump(this.value)"></select>
  <span id="label"></span>
  <span id="folder">{folder}</span>
</div>
<iframe id="frame"></iframe>

<script>
const files = {files_json};
let idx = 0;

const dd = document.getElementById('dropdown');
files.forEach((f, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = f;
  dd.appendChild(opt);
}});

function show(i) {{
  idx = i;
  document.getElementById('frame').src = '/__file__/' + encodeURIComponent(files[idx]);
  document.getElementById('label').textContent = (idx + 1) + ' / ' + files.length;
  document.getElementById('prev').disabled = idx === 0;
  document.getElementById('next').disabled = idx === files.length - 1;
  dd.value = idx;
}}

function go(delta) {{
  const ni = idx + delta;
  if (ni >= 0 && ni < files.length) show(ni);
}}

function jump(val) {{ show(parseInt(val)); }}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft')  go(-1);
  if (e.key === 'ArrowRight') go(1);
}});

show(0);
</script>
</body></html>"""


class ViewerHandler(http.server.BaseHTTPRequestHandler):
    folder = "."
    html_files = []

    def do_GET(self):
        path = urllib.parse.unquote(self.path)

        if path == "/" or path == "":
            content = build_viewer_html(self.html_files, self.folder).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)

        elif path.startswith("/__file__/"):
            fname = path[len("/__file__/"):]
            fpath = os.path.join(self.folder, fname)
            if os.path.isfile(fpath):
                with open(fpath, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", len(data))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404, f"Not found: {fname}")
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress per-request logging


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    folder = os.path.abspath(folder)

    html_files = sorted(f for f in os.listdir(folder) if f.endswith(".html"))
    if not html_files:
        print(f"No .html files found in {folder}")
        sys.exit(1)

    ViewerHandler.folder = folder
    ViewerHandler.html_files = html_files

    server = http.server.HTTPServer(("127.0.0.1", PORT), ViewerHandler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Serving {len(html_files)} files from {folder}")
    print(f"Open {url}  (Ctrl+C to stop)")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
