@echo off
setlocal enabledelayedexpansion
set VENV_DIR=venv

REM 仮想環境が存在しない場合は作成
if not exist "%VENV_DIR%" (
		python -m venv %VENV_DIR%
		echo 仮想環境を作成しました。
)


REM 仮想環境をアクティベート
set PYTHON=".\%VENV_DIR%\Scripts\activate"

REM requirements.txt から必要なモジュールをインストール
for /F "tokens=*" %%a in (requirements.txt) do (
		pip show %%a > NUL
		if errorlevel 1 (
		start /b pip install %%a
	)
)

REM srcディレクトリ内のwebui.pyを実行
python src\webui.py

pause
