#include <gtest/gtest.h>

#include <vector>

#include "seq/vasilev_s_gaus3x3/include/ops_seq.hpp"

namespace vasilev_s_gaus3x3_seq {

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

  vasilev_s_gaus3x3_seq::Gaus3x3Sequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), expected_validation);

  if (expected_validation) {
    taskSequential.pre_processing();
    taskSequential.run();
    taskSequential.post_processing();

    ASSERT_EQ(output_result, expected_result);
  }
}

}  // namespace vasilev_s_gaus3x3_seq

TEST(vasilev_s_gaus3x3_seq, three_five_simple_matrix) {
  vasilev_s_gaus3x3_seq::run_test(3, 5,
                                  {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0},
                                  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
}

TEST(vasilev_s_gaus3x3_seq, three_five_gradient_matrix) {
  vasilev_s_gaus3x3_seq::run_test(3, 5, {1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0},
                                  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 3.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
}

TEST(vasilev_s_gaus3x3_seq, validation_rows_less) {
  vasilev_s_gaus3x3_seq::run_test(2, 3, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, {}, false);
}

TEST(vasilev_s_gaus3x3_seq, validation_cols_less) {
  vasilev_s_gaus3x3_seq::run_test(3, 2, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, {}, false);
}

TEST(vasilev_s_gaus3x3_seq, validation_matrix_less) { vasilev_s_gaus3x3_seq::run_test(3, 3, {1.0}, {}, false); }
