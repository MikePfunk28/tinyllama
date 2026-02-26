@echo off
echo Training all experts sequentially with validation
echo ================================================

echo.
echo [1/4] Training REASONING expert...
python -u train_experts.py --expert reasoning --epochs 3
if exist experts\reasoning_best.pt (
    echo REASONING expert complete!
) else (
    echo REASONING expert FAILED!
)

echo.
echo [2/4] Training CONVERSATION expert...
python -u train_experts.py --expert conversation --epochs 3
if exist experts\conversation_best.pt (
    echo CONVERSATION expert complete!
) else (
    echo CONVERSATION expert FAILED!
)

echo.
echo [3/4] Training BUSINESS expert...
python -u train_experts.py --expert business --epochs 3
if exist experts\business_best.pt (
    echo BUSINESS expert complete!
) else (
    echo BUSINESS expert FAILED!
)

echo.
echo [4/4] Training KNOWLEDGE expert...
python -u train_experts.py --expert knowledge --epochs 3
if exist experts\knowledge_best.pt (
    echo KNOWLEDGE expert complete!
) else (
    echo KNOWLEDGE expert FAILED!
)

echo.
echo ================================================
echo Checking all experts...
python monitor_experts.py

echo.
echo Training MoE gate...
python moe_gating.py

echo.
echo DONE!
