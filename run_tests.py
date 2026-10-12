# -*- coding: utf-8 -*-
"""一键跑单测，不依赖 pytest。

用法：
    python run_tests.py
"""
import importlib
import os
import sys
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def run():
    tests_dir = os.path.join(BASE_DIR, "tests")
    total = passed = 0
    failures = []

    for fn in sorted(os.listdir(tests_dir)):
        if not fn.startswith("test_") or not fn.endswith(".py"):
            continue
        mod = importlib.import_module(f"tests.{fn[:-3]}")
        for name in dir(mod):
            if not name.startswith("test_"):
                continue
            func = getattr(mod, name)
            if not callable(func):
                continue
            total += 1
            try:
                func()
                passed += 1
                print(f"  OK  {fn}::{name}")
            except Exception:
                failures.append(f"{fn}::{name}")
                print(f"FAIL  {fn}::{name}")
                traceback.print_exc()

    print(f"\n通过 {passed}/{total} 个测试")
    if failures:
        print("失败：")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("全部通过 ✅")


if __name__ == "__main__":
    run()
