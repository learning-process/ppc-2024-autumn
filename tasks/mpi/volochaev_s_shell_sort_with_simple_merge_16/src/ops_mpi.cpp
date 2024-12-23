#include "mpi/volochaev_s_shell_sort_with_simple_merge_16/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <thread>
#include <vector>

namespace volochaev_s_shell_sort_with_simple_merge_16_mpi {
void sort(std::vector<int>& v);
std::vector<int> merge(const std::vector<int>& left, const std::vector<int>& right);
}  // namespace volochaev_s_shell_sort_with_simple_merge_16_mpi

bool volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_seq::pre_processing() {
  internal_order_test();

  int* x = reinterpret_cast<int*>(taskData->inputs[0]);
  size_ = static_cast<int>(taskData->inputs_count[0]);
  mas.resize(size_);

  std::copy(x, x + size_, mas.begin());

  return true;
}

bool volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_seq::validation() {
  internal_order_test();

  return static_cast<int>(taskData->inputs_count[0]) > 0;
}

bool volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_seq::run() {
  internal_order_test();

  int n = size_;
  int gap = n / 2;

  while (gap > 0) {
    for (int i = gap; i < n; ++i) {
      int temp = mas[i];
      int j = i;
      while (j >= gap && mas[j - gap] > temp) {
        mas[j] = mas[j - gap];
        j -= gap;
      }
      mas[j] = temp;
    }
    gap /= 2;
  }

  return true;
}

bool volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_seq::post_processing() {
  internal_order_test();

  int* ans = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(mas.begin(), mas.end(), ans);
  return true;
}

bool volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_mpi::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int* x = reinterpret_cast<int*>(taskData->inputs[0]);
    size_ = taskData->inputs_count[0];

    mas.resize(size_);

    std::copy(x, x + size_, mas.begin());
  }

  return true;
}

bool volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_mpi::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return static_cast<int>(taskData->inputs_count[0]) > 0;
  }

  return true;
}

void volochaev_s_shell_sort_with_simple_merge_16_mpi::sort(std::vector<int>& v) {
  int n = v.size();
  int gap = n / 2;

  while (gap > 0) {
    for (int i = gap; i < n; ++i) {
      int temp = v[i];
      int j = i;
      while (j >= gap && v[j - gap] > temp) {
        v[j] = v[j - gap];
        j -= gap;
      }
      v[j] = temp;
    }
    gap /= 2;
  }
}

std::vector<int> volochaev_s_shell_sort_with_simple_merge_16_mpi::merge(const std::vector<int>& left,
                                                                        const std::vector<int>& right) {
  std::vector<int> res;
  res.reserve(left.size() + right.size());
  std::merge(left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(res));

  return res;
}

bool volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_mpi::run() {
  internal_order_test();

  broadcast(world, size_, 0);
  std::vector<int> sizes;

  if (world.rank() == 0) {
    int delta = size_ / world.size();
    int ost = size_ % world.size();

    sizes.resize(world.size(), delta);
    for (int i = 0; i < ost; ++i) {
      ++sizes[i];
    }
  }

  broadcast(world, sizes, 0);

  local_input.resize(sizes[world.rank()]);
  scatterv(world, mas.data(), sizes, local_input.data(), 0);

  sort(local_input);

  if (world.rank() == 0) {
    mas = local_input;
    std::vector<int> data;

    for (int i = 1; i < world.size(); ++i) {
      world.recv(i, 0, data);
      mas = merge(mas, data);
    }
  } else {
    world.send(0, 0, local_input);
  }

  return true;
}

bool volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_mpi::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int* ans = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(mas.begin(), mas.end(), ans);
  }

  return true;
}
