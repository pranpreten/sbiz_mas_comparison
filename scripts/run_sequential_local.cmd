@echo off
setlocal
cd /d "%~dp0.."
rem Opt-in Solar answer generation; no LangSmith upload or grading.
set "LANGSMITH_TRACING=false"
set "LANGCHAIN_TRACING_V2=false"
set "MAX_MODEL_CALLS=20"
set "MAX_TOOL_CALLS=24"
set "MAX_ELAPSED_SECONDS=300"
"..\code\single_agent_test\.venv\Scripts\python.exe" -X utf8 -m mas_sequential.evaluate --local-run %*
set "sequential_exit_code=%errorlevel%"
endlocal & exit /b %sequential_exit_code%
