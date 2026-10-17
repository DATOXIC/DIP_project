@echo off
chcp 65001 >nul
cd /d "%~dp0"
REM chay.bat = nhan dien webcam | chay.bat demo = chay tren video mau (khong can camera) | chay.bat <lenh dip.py>
REM Mot lenh: tao moi truong ao, cai thu vien (chi lan dau), roi chay nhan dien webcam.
if not exist ".venv\Scripts\python.exe" (
  echo [1/2] Tao moi truong ao .venv ...
  py -3 -m venv .venv || (echo Can Python 3.12 hoac 3.13 - https://www.python.org/downloads/ & pause & exit /b 1)
)
if not exist ".venv\.da_cai" (
  echo [2/2] Cai thu vien, lan dau mat vai phut ...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip -q
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Cai that bai & pause & exit /b 1)
  echo ok> ".venv\.da_cai"
)
if "%~1"=="" (".venv\Scripts\python.exe" dip.py nhan-dien & goto :end)
if /i "%~1"=="demo" (".venv\Scripts\python.exe" pipeline.py --video demo\demo_clip.mp4 & goto :end)
".venv\Scripts\python.exe" dip.py %*
:end
pause
