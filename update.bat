@echo off
setlocal

REM Define the repository URL
set REPO_URL=https://github.com/ShiromiyaG/RVC-AI-Cover-Maker-UI

REM Navigate to the directory where the script is located
cd /d %~dp0

REM Loop through all directories except "env", logs" and "audio_files" and delete them
for /d %%D in (*) do (
    if /i not "%%D"=="env" if /i not "%%D"=="logs" if /i not "%%D"=="audio_files" (
        echo Deleting directory %%D
        rmdir /s /q "%%D"
    )
)

REM Loop through all files and delete them
for %%F in (*) do (
    if not "%%F"=="updater.bat" (
        echo Deleting file %%F
        del /q "%%F"
    )
)

REM Initialize a new git repository if it doesn't exist
if not exist .git (
    git init
    git remote add origin %REPO_URL%
)

REM Fetch the latest changes from the repository
git fetch origin

REM Reset the working directory to match the latest commit
git reset --hard origin/main

pause
endlocal