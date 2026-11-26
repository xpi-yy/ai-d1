#!/bin/bash
# 双击这个文件就能运行 D6，不经过 PyCharm
cd "$(dirname "$0")"

echo "========================================"
echo " D6：Embedding + 向量检索"
echo "========================================"
echo ""
echo "工作目录: $(pwd)"
echo "Python:   $(./.venv/bin/python -V 2>&1)"
echo ""

./.venv/bin/python d06_embedding.py

echo ""
echo "========================================"
echo " 运行结束。按回车关闭这个窗口。"
echo "========================================"
read
