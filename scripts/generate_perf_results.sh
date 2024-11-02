#!/usr/bin/env bash

mkdir -p build/perf_stat_dir
bash scripts/run_perf_collector.sh &> build/perf_stat_dir/perf_log.txt
if [ $? -ne 0 ]; then
    echo "scripts/run_perf_collector.sh failed. Displaying the log:"
    cat build/perf_stat_dir/perf_log.txt
    exit 1
else
    echo "Tests passed successfully! Generating reports..."
    # Run the Python script if the previous command succeeded
    python3 scripts/create_perf_table.py --input build/perf_stat_dir/perf_log.txt --output build/perf_stat_dir || exit 1
    echo "Reports have been generated!"
fi
