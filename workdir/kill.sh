# 先显示要杀的进程
pgrep -af "compile_worker|launcher"

# 确认后再杀（温和终止）
pgrep -f "compile_worker|launcher" | xargs -r kill

# 如果温和终止失败，改用强制
pgrep -f "compile_worker|launcher" | xargs -r kill -9
