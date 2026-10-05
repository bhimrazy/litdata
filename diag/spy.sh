#!/usr/bin/env bash
# run "$@" in background; if still running after $1 seconds, py-spy dump every python process
after=$1; shift
"$@" & pid=$!
for i in $(seq "$after"); do sleep 1; kill -0 $pid 2>/dev/null || { wait $pid; exit $?; }; done
echo "::warning::still running after ${after}s, dumping python stacks"
for p in $(pgrep -f python); do echo "===== pid $p: $(ps -o command= -p $p | cut -c1-200)"; sudo "$(which py-spy)" dump --pid $p 2>&1 | head -80; done
pkill -P $pid; kill $pid; pkill -f instr.py; exit 1
