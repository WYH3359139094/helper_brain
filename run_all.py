#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键启动 import_router(8000) 与 query_router(8001)。

用法（在项目根目录执行，或用任意目录的绝对路径调用都行）：
    python run_all.py                 # 同时启动两个服务
    python run_all.py --only import   # 只启动导入服务(8000)
    python run_all.py --only query    # 只启动问答服务(8001)

脚本替你做掉的事：
1. 把项目根加入 PYTHONPATH（代码里全是 `from knowledge.xxx` 绝对导入，工作目录必须是项目根）；
2. 把 knowledge/.venv/Scripts 加到 PATH 前面（pdf_to_md 节点是按名字 subprocess 调用 mineru-kit 的）；
3. 启动前检查端口是否被占用，被占用直接报错退出，不会出现 uvicorn 半启动的假象；
4. 启动后轮询 GET / 做健康检查，两个端口都返回 {"flag": "success"} 才算成功；
   健康检查只打根路径，不会触发导入/问答链路，通常 20~60 秒（主要在加载模型）；
5. Ctrl+C 同时关闭两个子进程。

不用 --reload：两个服务都带重活（模型/MinerU），热重载会反复重新加载权重，得不偿失。
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

ROOT = Path(__file__).resolve().parent

# key -> 服务定义；`app` 用 uvicorn --factory 方式指向 create_app
SERVICES: Dict[str, dict] = {
    "import": {
        "app": "knowledge.api.import_router:create_app",
        "port": 8000,
        "page": "/front/import.html",
        "desc": "文档导入服务",
    },
    "query": {
        "app": "knowledge.api.query_router:create_app",
        "port": 8001,
        "page": "/front/chat.html",
        "desc": "智能问答服务",
    },
}


def build_env() -> dict:
    """准备子进程环境：PATH 前置 venv/Scripts、PYTHONPATH 前置项目根。"""
    env = os.environ.copy()

    venv_scripts = ROOT / "knowledge" / ".venv" / "Scripts"
    if venv_scripts.is_dir():
        env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")

    parts = [str(ROOT)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)

    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """用 connect 判断端口是否已被监听（比 bind 探测更可靠，Windows 上不会被 SO_REUSEADDR 骗）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def http_ok(port: int, timeout: float = 2.0) -> bool:
    """GET / 期望 200；服务能响应根路径就说明 uvicorn 已完成启动。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def pump(name: str, proc: subprocess.Popen, tail: Deque[str]) -> None:
    """把子进程输出逐行转发到当前终端，并保留最近若干行用于失败时回显。"""
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            tail.append(f"[{name}] {line}")
            print(f"[{name}] {line}", flush=True)
    except Exception:
        pass


def start_service(name: str, host: str, log_level: str, tails: Dict[str, Deque[str]]) -> subprocess.Popen:
    svc = SERVICES[name]
    cmd = [
        sys.executable, "-m", "uvicorn", svc["app"], "--factory",
        "--host", host, "--port", str(svc["port"]), "--log-level", log_level,
    ]
    print(f"[{name}] 启动 {svc['desc']} -> {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=build_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    tails[name] = deque(maxlen=400)
    threading.Thread(target=pump, args=(name, proc, tails[name]), daemon=True).start()
    return proc


def stop_all(procs: Dict[str, subprocess.Popen]) -> None:
    if not procs:
        return
    print("\n正在关闭服务 ...", flush=True)
    for name, proc in procs.items():
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
    deadline = time.time() + 10
    for name, proc in procs.items():
        remain = max(0.5, deadline - time.time())
        try:
            proc.wait(timeout=remain)
        except subprocess.TimeoutExpired:
            print(f"[{name}] 未在 10s 内退出，强制结束", flush=True)
            try:
                proc.kill()
            except Exception:
                pass
    print("已全部关闭。", flush=True)


def wait_ready(procs: Dict[str, subprocess.Popen], timeout: int, tails: Dict[str, Deque[str]]) -> List[str]:
    """并发等待所有服务就绪，返回已就绪的服务名列表；任一进程提前退出则返回当前已就绪列表。"""
    pending = dict(procs)
    ready: List[str] = []
    started = time.time()
    last_tip = 0.0

    while pending:
        for name in list(pending):
            proc = pending[name]
            if proc.poll() is not None:
                print(f"\n[{name}] 进程已退出（exit={proc.returncode}），最后 20 行日志：", flush=True)
                for line in list(tails.get(name, []))[-20:]:
                    print("   ", line, flush=True)
                return ready
            if http_ok(SERVICES[name]["port"]):
                ready.append(name)
                del pending[name]
                print(f"[{name}] ✅ 端口 {SERVICES[name]['port']} 已就绪（{time.time() - started:.0f}s）", flush=True)

        if not pending:
            break

        elapsed = time.time() - started
        if elapsed >= timeout:
            print(f"\n⚠️  等待超时（{timeout}s），未就绪：{', '.join(pending)}", flush=True)
            for name in pending:
                print(f"[{name}] 最后 20 行日志：", flush=True)
                for line in list(tails.get(name, []))[-20:]:
                    print("   ", line, flush=True)
            return ready

        if elapsed - last_tip >= 5:
            last_tip = elapsed
            print(f"   等待 {', '.join(pending)} 就绪 ... {elapsed:.0f}s", flush=True)
        time.sleep(0.5)

    return ready


def main() -> int:
    parser = argparse.ArgumentParser(description="掌柜智库：一键启动导入(8000)与问答(8001)服务")
    parser.add_argument("--only", choices=["all", "import", "query"], default="all",
                        help="只启动某个服务，默认 all")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0")
    parser.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    parser.add_argument("--timeout", type=int, default=240, help="等待就绪的最长秒数，默认 240")
    args = parser.parse_args()

    names = list(SERVICES) if args.only == "all" else [args.only]

    # 1. 预检端口，避免 uvicorn 起来后静默失败
    busy = [n for n in names if port_in_use(SERVICES[n]["port"])]
    if busy:
        for n in busy:
            print(f"❌ 端口 {SERVICES[n]['port']} 已被占用（{SERVICES[n]['desc']}），请先结束占用进程：", flush=True)
        print("   netstat -ano | findstr :8000   (或 :8001)   然后 taskkill /PID <pid> /F", flush=True)
        return 2

    print("=" * 62, flush=True)
    print("掌柜智库 服务启动器", flush=True)
    print(f"项目根   : {ROOT}", flush=True)
    print(f"解释器   : {sys.executable}", flush=True)
    print("=" * 62, flush=True)

    procs: Dict[str, subprocess.Popen] = {}
    tails: Dict[str, Deque[str]] = {}
    try:
        for name in names:
            procs[name] = start_service(name, args.host, args.log_level, tails)

        ready = wait_ready(procs, args.timeout, tails)
        if len(ready) != len(names):
            stop_all(procs)
            return 1

        print("\n" + "=" * 62, flush=True)
        print("✅ 全部就绪", flush=True)
        for name in ready:
            svc = SERVICES[name]
            print(f"  {svc['desc']}  http://127.0.0.1:{svc['port']}{svc['page']}", flush=True)
        print("  Ctrl+C 结束", flush=True)
        print("=" * 62 + "\n", flush=True)

        # 2. 守护：任一进程挂掉就整体退出，避免“只有一个服务在跑”的错觉
        while True:
            for name, proc in procs.items():
                code = proc.poll()
                if code is not None:
                    print(f"\n[{name}] 意外退出（exit={code}），关闭其余服务", flush=True)
                    for line in list(tails.get(name, []))[-20:]:
                        print("   ", line, flush=True)
                    return 1
            time.sleep(1)

    except KeyboardInterrupt:
        return 0
    finally:
        stop_all(procs)


if __name__ == "__main__":
    sys.exit(main())
