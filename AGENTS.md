## 中文编码规则

本项目在 Windows 环境下开发，包含中文文档、中文注释和中文界面文本。

* 所有项目文件默认使用 `UTF-8` 编码。
* 读取或写入中文文件时，必须显式指定编码，不要依赖 Windows / PowerShell 默认编码。
* 优先使用 Python 读写中文文件：

```python
from pathlib import Path

path = Path("file.md")
text = path.read_text(encoding="utf-8")
path.write_text(text, encoding="utf-8", newline="\n")
```

* 不要用未指定编码的 `type`、`cat`、`Get-Content`、`Set-Content`、`Out-File`、`echo > file` 处理中文文件。
* 如果必须用 PowerShell，需显式指定 UTF-8：

```powershell
Get-Content "file.md" -Encoding UTF8
Set-Content "file.md" $content -Encoding UTF8
```

* 如果看到 `涓枃`、`ä¸­æ–‡`、`���` 等乱码，必须停止修改文件，先确认原文件编码，不要基于乱码内容重写。
* CSV 若用于 Excel 双击打开，可使用 `utf-8-sig`；其他情况默认 `utf-8`。
