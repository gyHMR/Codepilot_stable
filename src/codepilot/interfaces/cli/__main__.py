# 新手导读：命令行模块入口，通常只是把执行委托给同目录的 main.py。
# 关注点：这里不解析参数、不打开 session、不渲染界面；真正入口是 main.main()。

"""CLI 入口点：执行 ``python -m codepilot.interfaces.cli`` 时运行。

Python 的 ``-m`` 机制会寻找包下的 ``__main__.py``，因此这个文件只需要把
控制权交给 ``main()``。保持它很薄，可以让命令行入口和测试入口共用同一套逻辑。
"""

from .main import main

if __name__ == "__main__":
    # ``main`` 返回进程退出码；SystemExit 负责把它交还给操作系统。
    raise SystemExit(main())
