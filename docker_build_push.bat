@echo off
chcp 65001 > nul
echo =====================================================================
echo  AI MODEL DOCKER BUILD ^& PUBLISH TOOL (Music CoTMix)
echo =====================================================================
echo.

:: 1. Kiểm tra Docker
docker --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [LOI] Docker chua duoc cai dat hoac chua duoc them vao PATH!
    echo Vui long mo Docker Desktop truoc khi chay tool nay.
    pause
    exit /b 1
)

:: 2. Nhập thông tin Docker Hub
set /p DOCKER_USER="Nhap ten dang nhap Docker Hub cua ban: "
if "%DOCKER_USER%"=="" (
    echo [LOI] Ten dang nhap khong duoc de trong!
    pause
    exit /b 1
)

set IMAGE_NAME=%DOCKER_USER%/music-cotmix-ai
set IMAGE_TAG=latest

echo.
echo =====================================================================
echo  THONG TIN IMAGE SE BUILD:
echo  Repository: %IMAGE_NAME%
echo  Tag:        %IMAGE_TAG%
echo =====================================================================
echo.

:: 3. Đăng nhập Docker Hub
echo [Buoc 1/3] Dang nhap Docker Hub...
docker login
if %ERRORLEVEL% neq 0 (
    echo [LOI] Dang nhap Docker Hub that bai!
    pause
    exit /b 1
)

:: 4. Build Docker Image
echo.
echo [Buoc 2/3] Dang build Docker Image (qua trinh nay mat khoang 3-5 phut)...
docker build -t %IMAGE_NAME%:%IMAGE_TAG% -t %IMAGE_NAME%:v1.0.0 .
if %ERRORLEVEL% neq 0 (
    echo [LOI] Build Docker Image that bai!
    pause
    exit /b 1
)

:: 5. Push Image lên Docker Hub
echo.
echo [Buoc 3/3] Dang push Docker Image len Docker Hub cong khai...
docker push %IMAGE_NAME%:%IMAGE_TAG%
docker push %IMAGE_NAME%:v1.0.0
if %ERRORLEVEL% neq 0 (
    echo [LOI] Push Docker Image that bai!
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo  THANH CONG! Image cua ban da co mat tren Docker Hub tai:
echo  https://hub.docker.com/r/%IMAGE_NAME%
echo.
echo  Lenh de bat ky ai cung co the chay thu mo hinh cua ban:
echo  docker run -p 8000:8000 %IMAGE_NAME%:latest
echo =====================================================================
pause
