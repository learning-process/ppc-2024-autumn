#ifndef _AVG_MPI_HPP_
#define _AVG_MPI_HPP_

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

#ifndef RUN_TASK
#define RUN_TASK(task)              \
  ASSERT_TRUE((task).validation()); \
  (task).pre_processing();          \
  (task).run();                     \
  (task).post_processing();

#endif
namespace khasanyanov_k_average_vector_mpi {

template <class T>
std::vector<T> get_random_vector(size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<T> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = static_cast<T>(gen() % 1000 + (gen() % 100) / 100.0);
  }
  return vec;
}

template <class InType, class OutType>
std::shared_ptr<ppc::core::TaskData> create_task_data(std::vector<InType>& in, std::vector<OutType>& out) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  return taskData;
}

//=========================================sequential=========================================

template <class In, class Out>
class AvgVectorMPITaskSequential : public ppc::core::Task {
  std::vector<In> input_;
  Out avg = 0.0;

 public:
  explicit AvgVectorMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<In, Out>::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<In, Out>::pre_processing() {
  internal_order_test();
  input_ = std::vector<In>(taskData->inputs_count[0]);
  auto* tmp = reinterpret_cast<In*>(taskData->inputs[0]);
  std::copy(tmp, tmp + taskData->inputs_count[0], std::back_inserter(input_));
  avg = 0.0;
  return true;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<In, Out>::run() {
  internal_order_test();
  avg = static_cast<Out>(std::accumulate(input_.begin(), input_.end(), 0.0, std::plus()));
  avg /= static_cast<Out>(taskData->inputs_count[0]);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  return true;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<In, Out>::post_processing() {
  internal_order_test();
  reinterpret_cast<Out*>(taskData->outputs[0])[0] = avg;
  return true;
}

//=========================================parallel=========================================

namespace mpi = boost::mpi;
template <class In, class Out>
class AvgVectorMPITaskParallel : public ppc::core::Task {
  std::vector<In> input_, local_input_;
  Out avg = 0.0;
  mpi::communicator world;

 public:
  explicit AvgVectorMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static std::pair<std::vector<int>, std::vector<int>> displacement(size_t, size_t);
};

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<In, Out>::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0;
  }
  return true;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<In, Out>::pre_processing() {
  internal_order_test();
  size_t input_size;
  if (world.rank() == 0) {
    input_size = taskData->inputs_count[0];
  }

  mpi::broadcast(world, input_size, 0);

  std::pair<std::vector<int>, std::vector<int>> disp = displacement(input_size, world.size());
  auto& displacements = disp.second;
  auto& sizes = disp.first;
  if (world.rank() == 0) {
    input_ = std::vector<In>(taskData->inputs_count[0]);
    auto* tmp = reinterpret_cast<In*>(taskData->inputs[0]);

    input_.clear();
    std::copy(tmp, tmp + taskData->inputs_count[0], std::back_inserter(input_));

    local_input_.resize(sizes[0]);
    mpi::scatterv(world, input_, sizes, displacements, local_input_.data(), sizes[0], 0);

  } else {
    local_input_.resize(sizes[world.rank()]);
    mpi::scatterv(world, local_input_.data(), sizes[world.rank()], 0);
  }
  avg = 0.0;
  return true;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<In, Out>::run() {
  internal_order_test();
  Out local_sum{};
  local_sum = static_cast<Out>(std::accumulate(local_input_.begin(), local_input_.end(), 0.0, std::plus()));
  mpi::reduce(world, local_sum, avg, std::plus(), 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  return true;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<In, Out>::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<Out*>(taskData->outputs[0])[0] = avg / input_.size();
  }
  return true;
}

template <class In, class Out>
std::pair<std::vector<int>, std::vector<int>>
khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<In, Out>::displacement(size_t input_size, size_t n) {
  const size_t capacity = n;
  size_t count = input_size / capacity;
  size_t mod = input_size % capacity;
  std::vector<int> sizes(capacity, count);
  std::transform(sizes.cbegin(), sizes.cbegin() + mod, sizes.begin(), [](auto i) { return i + 1; });
  std::vector<int> disp(capacity);
  disp[0] = 0;
  size_t i = 0;
  std::generate(disp.begin() + 1, disp.end(), [&]() {
    ++i;
    return disp[i - 1] + sizes[i - 1];
  });
  return {sizes, disp};
}

}  //  namespace khasanyanov_k_average_vector_mpi

#endif  // !_AVG_MPI_HPP_
