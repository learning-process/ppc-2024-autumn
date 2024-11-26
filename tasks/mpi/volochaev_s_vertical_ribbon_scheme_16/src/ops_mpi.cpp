#include "mpi/volochaev_s_vertical_ribbon_scheme_16/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_seq::pre_processing() {
  internal_order_test();

  input_A = reinterpret_cast<int*>(taskData->inputs[0]);
  input_B = reinterpret_cast<int*>(taskData->inputs[1]);
  int c = taskData->inputs_count[0];
  m = taskData->inputs_count[1];
  n = c / m;
  res.assign(n, 0);

  return true;
}

bool volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_seq::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 1 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
}

bool volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_seq::run() {
  internal_order_test();

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      res[j] += input_A[i * n + j] * input_B[i];
    }
  }

  return true;
}

bool volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_seq::post_processing() {
  internal_order_test();

  int* ans = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res.begin(), res.end(), ans);

  return true;
}

bool volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_mpi::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    if (!taskData || taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr ||
        taskData->outputs[0] == nullptr) {
      return false;
    }

    int* input_A = reinterpret_cast<int*>(taskData->inputs[0]);
    int* input_B = reinterpret_cast<int*>(taskData->inputs[1]);

    int c = taskData->inputs_count[0];
    m = taskData->inputs_count[1];
    n = c / m;

    input_A1.assign(input_A, input_A + c);
    input_B1.assign(input_B, input_B + m);
    res.resize(n, 0);

    distribution.resize(world.size(), 0);
    displacement.resize(world.size(), -1);

    if (world.size() > m) {
      for (int i = 0; i < m; ++i) {
        distribution[i] = n;
        displacement[i] = i * n;
      }
    } else {
      int m_per_proc = m / world.size();
      int ost = m % world.size();

      int offset = 0;
      for (int i = 0; i < world.size(); ++i) {
        if (ost > 0) {
          distribution[i] = (m_per_proc + 1) * n;
          --ost;
        } else {
          distribution[i] = m_per_proc * n;
        }
        displacement[i] = offset;
        offset += distribution[i];
      }
    }
  }

  if (world.rank() != 0) {
    input_A1.resize(n * m, 0);
    input_B1.resize(m, 0);
    res.resize(n, 0);
  }

  return true;
}

bool volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_mpi::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->inputs_count[0] > 1 && taskData->inputs_count[1] > 0 &&
           taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
  }
  return true;
}

bool volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_mpi::run() {
  internal_order_test();

  boost::mpi::broadcast(world, m, 0);
  boost::mpi::broadcast(world, n, 0);
  boost::mpi::broadcast(world, distribution, 0);
  boost::mpi::broadcast(world, displacement, 0);
  boost::mpi::broadcast(world, input_A1, 0);
  boost::mpi::broadcast(world, input_B1, 0);

  int local_start_col = displacement[world.rank()] / n;
  int local_m = distribution[world.rank()] / n;
  std::vector<int> local_res(n, 0);

  for (int i = 0; i < local_m; ++i) {
    for (int j = 0; j < n; ++j) {
      int glob_col = local_start_col + i;
      int A = input_A1[glob_col * n + j];
      int B = input_B1[glob_col];
      local_res[j] += A * B;
    }
  }

  boost::mpi::reduce(world, local_res.data(), n, res.data(), std::plus<>(), 0);

  return true;
}

bool volochaev_s_vertical_ribbon_scheme_16_mpi::Lab2_16_mpi::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int* ans = reinterpret_cast<int*>(taskData->outputs[0]);
    for (int i = 0; i < res.size(); ++i) {
      ans[i] = res[i];
    }
  }

  return true;
}
