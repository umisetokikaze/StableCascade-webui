#!/bin/bash

# 仮想環境ディレクトリの名前
VENV_DIR="venv"

# 仮想環境が存在しない場合は作成
if [ ! -d "$VENV_DIR" ]; then
    python -m venv $VENV_DIR
    echo "仮想環境を作成しました。"
fi

# 仮想環境をアクティベート
source ./$VENV_DIR/bin/activate

# requirements.txt から必要なモジュールをインストール
pip install -r requirements.txt

# srcディレクトリ内のwebui.pyを実行
python src/webui.py "$@"