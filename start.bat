@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   RAG 网页版启动
echo ============================================
echo 正在启动，浏览器会自动打开 http://localhost:8501
echo 首次建索引会下载模型，请耐心等待
echo 停止程序：在这个窗口按 Ctrl+C
echo.
streamlit run app.py
echo.
echo ============ 程序已停止 ============
echo 如果上面有红色/英文报错，截图发给我
pause
