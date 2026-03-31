#!/usr/bin/env python3
"""
PPT Master - slide_state 项目桥接工具

用途：
1. 将现有项目的 svg_output/ 或 svg_final/ 收敛为 slide_state.json
2. 从 slide_state.json 重新渲染 svg_output/，继续走 finalize/export 链路
3. 为当前“Executor 先产出 SVG”工作流补上正式的 slide_state 真相源

用法:
    python3 tools/slide_state_bridge.py capture <project_path> [--source-dir svg_output] [--state-file slide_state.json]
    python3 tools/slide_state_bridge.py render <project_path> [--output-dir svg_output] [--state-file slide_state.json]
    python3 tools/slide_state_bridge.py sync <project_path> [--source-dir svg_output] [--output-dir svg_output] [--state-file slide_state.json]
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EDITOR_DIR = REPO_ROOT / 'editor'
CLI_TS_PATH = EDITOR_DIR / 'src' / 'cli.ts'
CLI_JS_PATH = EDITOR_DIR / 'dist' / 'cli.js'


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] in {'-h', '--help'}:
        print(__doc__)
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    command = sys.argv[1]
    project_path = sys.argv[2]
    passthrough_args = sys.argv[3:]

    command_mapping = {
        'capture': 'capture-project',
        'render': 'render-project',
        'sync': 'sync-project',
    }
    cli_command = command_mapping.get(command)
    if cli_command is None:
        print(f"错误: 未知命令 '{command}'")
        print(__doc__)
        sys.exit(1)

    run_editor_cli([cli_command, project_path, *passthrough_args])


def run_editor_cli(args: list[str]) -> None:
    bun = shutil.which('bun')
    if bun:
        completed = subprocess.run(
            [bun, 'run', str(CLI_TS_PATH), *args],
            cwd=REPO_ROOT,
            check=False,
        )
        sys.exit(completed.returncode)

    node = shutil.which('node')
    npm = shutil.which('npm')
    if not node or not npm:
        raise SystemExit('错误: 需要 Bun，或同时安装 Node.js + npm 才能运行 slide_state bridge')

    ensure_compiled_cli(npm)
    completed = subprocess.run(
        [node, str(CLI_JS_PATH), *args],
        cwd=REPO_ROOT,
        check=False,
    )
    sys.exit(completed.returncode)


def ensure_compiled_cli(npm: str) -> None:
    print('[slide_state_bridge] 当前使用 Node fallback，正在执行 `npm run build` ...', flush=True)
    completed = subprocess.run(
        [npm, 'run', 'build'],
        cwd=EDITOR_DIR,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
