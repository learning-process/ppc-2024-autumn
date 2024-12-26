#ifndef _AVG_SEQ_HPP_
#define _AVG_SEQ_HPP_

#include <gtest/gtest.h>

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
namespace khasanyanov_k_average_vector_seq {

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
class AvgVectorSEQTaskSequential : public ppc::core::Task {
  std::vector<In> input_;
  Out avg = 0.0;

 public:
  explicit AvgVectorSEQTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};

template <class In, class Out>
bool khasanyanov_k_average_vector_seq::AvgVectorSEQTaskSequential<In, Out>::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_seq::AvgVectorSEQTaskSequential<In, Out>::pre_processing() {
  internal_order_test();
  input_ = std::vector<In>(taskData->inputs_count[0]);
  auto* tmp = reinterpret_cast<In*>(taskData->inputs[0]);
  std::copy(tmp, tmp + taskData->inputs_count[0], std::back_inserter(input_));
  avg = 0.0;
  return true;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_seq::AvgVectorSEQTaskSequential<In, Out>::run() {
  internal_order_test();
  avg = static_cast<Out>(std::accumulate(input_.begin(), input_.end(), 0.0, std::plus()));
  avg /= static_cast<Out>(taskData->inputs_count[0]);
  // std::this_thread::sleep_for(std::chrono::milliseconds(5));
  return true;
}

template <class In, class Out>
bool khasanyanov_k_average_vector_seq::AvgVectorSEQTaskSequential<In, Out>::post_processing() {
  internal_order_test();
  reinterpret_cast<Out*>(taskData->outputs[0])[0] = avg;
  return true;
}

}  // namespace khasanyanov_k_average_vector_seq

#endif  // !_AVG_MPI_HPP_
