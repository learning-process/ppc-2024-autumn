#include <gtest/gtest.h>

#include <vector>

#include "seq/plekhanov_d_verticalgaus/include/ops_seq.hpp"

namespace plekhanov_d_verticalgaus_seq {

void run_test(int num_rows, int num_cols, const std::vector<double>& input_matrix,
              const std::vector<double>& expected_result, bool expected_validation = true) {
  std::vector<double> output_result(num_rows * num_cols, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  plekhanov_d_verticalgaus_seq::VerticalGausSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), expected_validation);

  if (expected_validation) {
    taskSequential.pre_processing();
    taskSequential.run();
    taskSequential.post_processing();

    ASSERT_EQ(output_result, expected_result);
  }
}

}  // namespace plekhanov_d_verticalgaus_seq

TEST(plekhanov_d_verticalgaus_seq, Matrix1x1) { plekhanov_d_verticalgaus_seq::run_test(1, 1, {1}, {0}, false); }

TEST(plekhanov_d_verticalgaus_seq, Matrix_3x5) {
  plekhanov_d_verticalgaus_seq::run_test(3, 5,
                                  {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0},
                                  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
}

TEST(plekhanov_d_verticalgaus_seq, Matrix_5x3) {
  plekhanov_d_verticalgaus_seq::run_test(5, 3, 
                                  {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0},
                                  {0, 0, 0, 0, 5, 0, 0, 8, 0, 0, 11, 0, 0, 0, 0});
}

TEST(plekhanov_d_verticalgaus_seq, Matrix_4x4) {
  plekhanov_d_verticalgaus_seq::run_test(4, 4,
                                  {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                                  {0, 0, 0, 0, 0, 6, 7, 0, 0, 10, 11, 0, 0, 0, 0, 0});
}

TEST(plekhanov_d_verticalgaus_seq, Matrix0x0) {
  plekhanov_d_verticalgaus_seq::run_test(0, 0, {}, {}, false);
}