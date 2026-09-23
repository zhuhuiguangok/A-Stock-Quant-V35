# -*- coding: utf-8 -*-
"""Django 服务守护进程（替代 run_server_forever.bat）。

与 bat 的关键差异：子进程 stdout/stderr 直接重定向到日志文件句柄，
**完全不继承控制台**——控制台窗口被关闭也不会导致子进程拿到死句柄
而 fail-fast（0xC0000409 崩溃循环的根因）。

由 run_server.bat（双击后立即隐藏后台运行）或计划任务启动：
    powershell -Command "Start-Process -FilePath <python> -ArgumentList \
        'run_server_forever.py' -WorkingDirectory <项目根> -WindowStyle Hidden"
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "logs" / "server_forever.log"
PORT = "18935"


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    LOG.parent.mkdir(exist_ok=True)
    while True:
        with open(LOG, "ab") as fh:
            fh.write(f"\n[{stamp()}] supervisor: starting server\n".encode())
            try:
                proc = subprocess.run(
                    [sys.executable, "-u", "manage.py", "runserver",
                     f"127.0.0.1:{PORT}", "--noreload"],
                    cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                    # 不传任何控制台相关句柄；Windows 下默认继承父进程（本身无窗口）
                )
                code = proc.returncode
            except Exception as exc:
                code = f"supervisor-error: {exc}"
        with open(LOG, "ab") as fh:
            fh.write(f"[{stamp()}] supervisor: server exited code={code}, restart in 5s\n".encode())
        time.sleep(5)


if __name__ == "__main__":
    main()
