#include "mpi/varfolomeev_g_quick_sort_simple_merge/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

bool varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  res = input_;
  return true;
}

bool varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count.size() == 1 && taskData->inputs_count.size() == 1;
}

bool varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential::run() {
  internal_order_test();
  quickSort(res);
  return true;
}

bool varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res.begin(), res.end(), output_ptr);
  return true;
}

bool varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  // Получаем входные данные на корневом процессе (rank 0)
  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);
  }
  return true;
}

bool varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->outputs_count.size() == 1 && taskData->inputs_count.size() == 1 && world.size() > 0 &&
            world.rank() < world.size());
  }
  return true;
}

bool varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  boost::mpi::broadcast(world, input_, 0);
  int local_size = input_.size() / world.size();
  int tail = input_.size() % world.size();

  std::vector<int> distribution(world.size(), local_size);
  for (int i = 0; i < tail; ++i) {
    distribution[i]++;
  }

  std::vector<int> begins(world.size());
  begins[0] = 0;
  for (int i = 1; i < world.size(); ++i) {
    begins[i] = begins[i - 1] + distribution[i - 1];
  }

  local_input_.resize(distribution[world.rank()]);
  boost::mpi::scatterv(world, input_.data(), distribution, begins, local_input_.data(), distribution[world.rank()], 0);

  local_input_ = quickSortRecursive(local_input_);

  std::vector<int> allData;
  if (world.rank() == 0) {
    allData.resize(input_.size());
  }
  boost::mpi::gatherv(world, local_input_.data(), distribution[world.rank()], allData.data(), distribution, begins, 0);
  if (world.rank() == 0) {
    res.assign(allData.begin(), allData.begin() + distribution[0]);
    for (int i = 1; i < world.size(); ++i) {
      std::vector<int> right(allData.begin() + begins[i], allData.begin() + begins[i] + distribution[i]);
      res = merge(res, right);
    }
  }
  return true;
}

bool varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < static_cast<int>(res.size()); ++i) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}
