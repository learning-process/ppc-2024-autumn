// Copyright 2023 Nesterov Alexander
#pragma once

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <limits>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/ermolaev_v_allreduce_my/include/allreduce.hpp"
#include "mpi/ermolaev_v_allreduce_my/include/shared_ptr_array.hpp"

namespace ermolaev_v_allreduce_mpi {

template <typename _T, typename _S = uint32_t>
class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<_T>> input_, res_;
};

template <typename _T, typename _S = uint32_t>
class TemplateTestTaskParallel : public ppc::core::Task {
 public:
  explicit TemplateTestTaskParallel(
      std::shared_ptr<ppc::core::TaskData> taskData_,
      std::function<void(const boost::mpi::communicator&, const _T*, int, _T*, boost::mpi::minimum<_T>)> min_func_,
      std::function<void(const boost::mpi::communicator&, const _T*, int, _T*, boost::mpi::maximum<_T>)> max_func_)
      : Task(std::move(taskData_)), min_allreduce_(std::move(min_func_)), max_allreduce_(std::move(max_func_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 protected:
  std::vector<std::vector<_T>> input_, local_input_, res_;

  const std::function<void(const boost::mpi::communicator&, const _T*, int, _T*, boost::mpi::minimum<_T>)>
      min_allreduce_;
  const std::function<void(const boost::mpi::communicator&, const _T*, int, _T*, boost::mpi::maximum<_T>)>
      max_allreduce_;
  boost::mpi::communicator world;
};

template <typename _T, typename _Op>
void func(const boost::mpi::communicator& world, const _T* src, int n, _T* dst, _Op op) {
  boost::mpi::all_reduce<_T, _Op>(world, src, n, dst, op);
};

template <typename _T>
class DefaultAllReduceTask : public TemplateTestTaskParallel<_T> {
 public:
  DefaultAllReduceTask(std::shared_ptr<ppc::core::TaskData> taskData_)
      : TemplateTestTaskParallel<_T>(taskData_, func<_T, boost::mpi::minimum<_T>>, func<_T, boost::mpi::maximum<_T>>) {}
};

template <typename _T>
class MyAllReduceTask : public TemplateTestTaskParallel<_T> {
 public:
  MyAllReduceTask(std::shared_ptr<ppc::core::TaskData> taskData_)
      : TemplateTestTaskParallel<_T>(taskData_, ermolaev_v_allreduce_mpi::allreduce<_T, boost::mpi::minimum<_T>>,
                                     ermolaev_v_allreduce_mpi::allreduce<_T, boost::mpi::maximum<_T>>) {}
};

}  // namespace ermolaev_v_allreduce_mpi

// TestMPITaskSequential
template <typename _T, typename _S>
bool ermolaev_v_allreduce_mpi::TestMPITaskSequential<_T, _S>::pre_processing() {
  internal_order_test();

  input_.resize(taskData->inputs_count[0], std::vector<_T>(taskData->inputs_count[1]));

  auto* ptr = reinterpret_cast<ermolaev_v_allreduce_mpi::shared_ptr_array<_T>*>(taskData->inputs[0]);
  for (_S i = 0; i < input_.size(); i++) {
    for (_S j = 0; j < input_[i].size(); j++) {
      input_[i][j] = ptr[i][j];
    }
  }

  res_.resize(taskData->inputs_count[0], std::vector<_T>(taskData->inputs_count[1]));

  return true;
}

template <typename _T, typename _S>
bool ermolaev_v_allreduce_mpi::TestMPITaskSequential<_T, _S>::validation() {
  internal_order_test();

  return (taskData->inputs_count == taskData->outputs_count && !taskData->inputs.empty() && !taskData->outputs.empty());
}

template <typename _T, typename _S>
bool ermolaev_v_allreduce_mpi::TestMPITaskSequential<_T, _S>::run() {
  internal_order_test();

  for (_S i = 0; i < input_.size(); i++) {
    const auto [min, max] = std::minmax_element(input_[i].begin(), input_[i].end());
    for (_S j = 0; j < input_[i].size(); j++) {
      if ((*max - *min) <= std::numeric_limits<_T>::epsilon())
        res_[i][j] = 0;
      else
        res_[i][j] = (input_[i][j] - *min) / (*max - *min);
    }
  }

  return true;
}

template <typename _T, typename _S>
bool ermolaev_v_allreduce_mpi::TestMPITaskSequential<_T, _S>::post_processing() {
  internal_order_test();

  auto* ptr = reinterpret_cast<ermolaev_v_allreduce_mpi::shared_ptr_array<_T>*>(taskData->outputs[0]);
  for (_S i = 0; i < res_.size(); i++) {
    std::copy(res_[i].begin(), res_[i].end(), ptr[i].get());
  }

  return true;
}
// TestMPITaskSequential

// TemplateTestTaskParallel
template <typename _T, typename _S>
bool ermolaev_v_allreduce_mpi::TemplateTestTaskParallel<_T, _S>::validation() {
  internal_order_test();

  return world.rank() != 0 ||
         (taskData->inputs_count == taskData->outputs_count && !taskData->inputs.empty() && !taskData->outputs.empty());
}

template <typename _T, typename _S>
bool ermolaev_v_allreduce_mpi::TemplateTestTaskParallel<_T, _S>::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    input_.resize(taskData->inputs_count[1], std::vector<_T>(taskData->inputs_count[0]));

    auto* ptr = reinterpret_cast<ermolaev_v_allreduce_mpi::shared_ptr_array<_T>*>(taskData->inputs[0]);
    for (_S j = 0; j < taskData->inputs_count[1]; j++) {
      for (_S i = 0; i < taskData->inputs_count[0]; i++) {
        input_[j][i] = ptr[i][j];
      }
    }

    res_.resize(taskData->inputs_count[1], std::vector<_T>(taskData->inputs_count[0]));
  }

  return true;
}

template <typename _T, typename _S>
bool ermolaev_v_allreduce_mpi::TemplateTestTaskParallel<_T, _S>::run() {
  internal_order_test();

  _S segment = 0;
  _S over = 0;
  _S rows = 0;
  if (world.rank() == 0) {
    segment = taskData->inputs_count[1] / world.size();
    over = taskData->inputs_count[1] % world.size();
    rows = taskData->inputs_count[0];
  }

  broadcast(world, segment, 0);
  broadcast(world, over, 0);
  broadcast(world, rows, 0);

  std::vector<int> size(world.size(), (int)segment);
  for (_S i = 0; i < over; i++) size[world.size() - i - 1]++;

  local_input_.resize(size[world.rank()]);
  scatterv(world, input_, size, local_input_.data(), 0);

  std::vector<_T> min_by_rows(rows, std::numeric_limits<_T>::max());
  std::vector<_T> local_min_by_rows(rows, std::numeric_limits<_T>::max());

  std::vector<_T> max_by_rows(rows, -std::numeric_limits<_T>::max());
  std::vector<_T> local_max_by_rows(rows, -std::numeric_limits<_T>::max());

  for (_S j = 0; j < local_input_.size(); j++) {
    for (_S i = 0; i < local_input_[j].size(); i++) {
      if (local_input_[j][i] < local_min_by_rows[i]) local_min_by_rows[i] = local_input_[j][i];
      if (local_input_[j][i] > local_max_by_rows[i]) local_max_by_rows[i] = local_input_[j][i];
    }
  }

  // Main funcs
  min_allreduce_(world, local_min_by_rows.data(), local_min_by_rows.size(), min_by_rows.data(),
                 boost::mpi::minimum<_T>());
  max_allreduce_(world, local_max_by_rows.data(), local_max_by_rows.size(), max_by_rows.data(),
                 boost::mpi::maximum<_T>());
  //

  auto local_res(local_input_);
  for (_S j = 0; j < local_res.size(); j++) {
    for (_S i = 0; i < local_res[j].size(); i++) {
      if ((max_by_rows[i] - min_by_rows[i]) <= std::numeric_limits<_T>::epsilon())
        local_res[j][i] = 0;
      else
        local_res[j][i] = (local_res[j][i] - min_by_rows[i]) / (max_by_rows[i] - min_by_rows[i]);
    }
  }

  gatherv(world, local_res, res_.data(), size, 0);

  return true;
}

template <typename _T, typename _S>
bool ermolaev_v_allreduce_mpi::TemplateTestTaskParallel<_T, _S>::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* ptr = reinterpret_cast<ermolaev_v_allreduce_mpi::shared_ptr_array<_T>*>(taskData->outputs[0]);
    for (_S j = 0; j < res_.size(); j++) {
      for (_S i = 0; i < res_[j].size(); i++) {
        ptr[i][j] = res_[j][i];
      }
    }
  }

  return true;
}
// TemplateTestTaskParallel
