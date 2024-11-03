#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/nasedkin_e_matrix_column_max_value/include/ops_seq.hpp"

TEST(SeqMatrixColumnMaxPerfTest, PerformanceTest) {
  boost::mpi::timer timer;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input(1000000, 1);  // большой вектор для теста
  int output = 0;

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
  taskData->outputs_count.emplace_back(1);

  auto task = std::make_shared<nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxTaskSequential>(taskData, "max");

  task->pre_processing();
  task->run();
  task->post_processing();

  ASSERT_EQ(output, 1);

  double elapsed_time = timer.elapsed();
  std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}