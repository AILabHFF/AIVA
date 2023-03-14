@echo off

REM Check if virtual environment exists
if not exist "venv" (
  REM Create virtual environment
  echo Creating virtual environment...
  python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install required packages
echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM Run Flask app
echo Running Flask app...
python app.py

REM Deactivate virtual environment
echo Deactivating virtual environment...
deactivate
