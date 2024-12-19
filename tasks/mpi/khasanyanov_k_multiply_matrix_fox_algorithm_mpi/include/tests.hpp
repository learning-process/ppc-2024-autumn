#ifndef _TESTS_HPP_
#define _TESTS_HPP_

#include <gtest/gtest.h>

#include "../modules/core/task/include/task.hpp"
#include "matrix.hpp"

namespace khasanyanov_k_fox_algorithm {

#define RUN_TASK(task)              \
  ASSERT_TRUE((task).validation()); \
  (task).pre_processing();          \
  (task).run();                     \
  (task).post_processing();

template <typename DataType>
std::shared_ptr<ppc::core::TaskData> create_task_data(matrix<DataType>& A, matrix<DataType>& B, matrix<DataType>& C) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(*A));
  taskData->inputs_count.emplace_back(static_cast<uint32_t>(A.rows));
  taskData->inputs_count.emplace_back(static_cast<uint32_t>(A.columns));
  taskData->inputs_count.emplace_back(static_cast<uint32_t>(A.size()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(*B));
  taskData->inputs_count.emplace_back(static_cast<uint32_t>(B.rows));
  taskData->inputs_count.emplace_back(static_cast<uint32_t>(B.columns));
  taskData->inputs_count.emplace_back(static_cast<uint32_t>(B.size()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(*C));
  taskData->outputs_count.emplace_back(static_cast<uint32_t>(C.rows));
  taskData->outputs_count.emplace_back(static_cast<uint32_t>(C.columns));
  taskData->outputs_count.emplace_back(static_cast<uint32_t>(C.size()));
  return taskData;
}

}  // namespace khasanyanov_k_fox_algorithm

#endif