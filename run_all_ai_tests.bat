@echo off
chcp 65001 > nul
cls
echo ===============================================================================
echo       HE THONG KIEM THU VA TAN CONG AI TOAN DIEN (MUSIC COTMIX AI)
echo                  Chuyen vien Kiem thu AI (QA & Red Teaming)
echo ===============================================================================
echo.
echo [1/3] Kiem tra moi truong Python va Pytest...
python --version
if errorlevel 1 (
    echo [LOI] Khong tim thay Python! Vui long cai dat Python 3.10+ va them vao PATH.
    pause
    exit /b 1
)
echo.
echo ===============================================================================
echo [2/3] DANG CHAY 10 TEST CASE KIEM THU HE THONG AI (API & SYSTEM)...
echo       (Bao gom: Automation test, AI Accuracy, Confidence Gate, Mock test)
echo ===============================================================================
python -m pytest tests/test_01_automation_health.py tests/test_02_ai_accuracy.py tests/test_03_ai_confidence_gate.py tests/test_04_ai_domain_shift.py tests/test_05_automation_latency.py tests/test_06_automation_input_validation.py tests/test_07_mock_checkpoint_failure.py tests/test_08_mock_missing_samples.py tests/test_09_ai_xai_explainability.py tests/test_10_automation_api_contract.py -v --tb=short
if errorlevel 1 (
    echo [CANH BAO] Mot so test case he thong can kiem tra lai!
) else (
    echo.
    echo [THANH CONG] TAT CA 10 TEST CASE HE THONG AI DA VUOT QUA 100%%!
)
echo.
echo ===============================================================================
echo [3/3] DANG CHAY 10 TEST CASE TAN CONG / KIEM THU AN TOAN BAO MAT AI...
echo       (Bao gom: Adversarial ML, Injection, DoS, Model Inversion, Supply Chain)
echo ===============================================================================
python -m pytest tests/attacks/ -v --tb=short
if errorlevel 1 (
    echo [CANH BAO] Phat hien cac lo hong can khac phuc!
) else (
    echo.
    echo [THANH CONG] TAT CA 10 KIEM THU TAN CONG DA DUOC DANH GIA DAY DU!
)
echo.
echo ===============================================================================
echo                           HOAN TAT TOAN BO KIEM THU!
echo ===============================================================================
pause
