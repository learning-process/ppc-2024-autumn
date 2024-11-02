#!/bin/bash

if [[ -z "$ASAN_RUN" ]]; then
  if [[ $OSTYPE == "linux-gnu" ]]; then
    mpirun --oversubscribe -np 4 ./build/bin/sample_mpi
    mpirun --oversubscribe -np 4 ./build/bin/sample_mpi_boost
  elif [[ $OSTYPE == "darwin"* ]]; then
    mpirun -np 2 ./build/bin/sample_mpi
    mpirun -np 2 ./build/bin/sample_mpi_boost
  fi
fi
./build/bin/sample_omp
./build/bin/sample_stl
./build/bin/sample_tbb

# separate tests for debug
for test_item in $(./build/bin/mpi_perf_tests --gtest_list_tests | awk '/\./{ SUITE=$1 }  /  / { print SUITE $1 }')
do
  if [[ -z "$ASAN_RUN" ]]; then
    if [[ $OSTYPE == "linux-gnu" ]]; then
      mpirun --oversubscribe -np 4 ./build/bin/mpi_perf_tests --gtest_filter="$test_item"
    elif [[ $OSTYPE == "darwin"* ]]; then
      mpirun -np 2 ./build/bin/mpi_perf_tests --gtest_filter="$test_item"
    fi
  fi
done

./build/bin/omp_perf_tests
./build/bin/seq_perf_tests
./build/bin/stl_perf_tests
./build/bin/tbb_perf_tests
