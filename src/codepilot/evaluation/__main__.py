from __future__ import annotations

# 新手导读：命令行模块入口，通常只是把执行委托给同目录的 main/cli。
# 关注点：新手不必深挖这里，顺着导入跳到真正入口即可。

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
