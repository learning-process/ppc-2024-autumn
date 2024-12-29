#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/plekhanov_d_verticalgaus/include/ops_seq.hpp"

namespace plekhanov_d_verticalgaus_seq {

std::vector<double> getRandomImage(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(0, 255);
  std::vector<double> img(sz);
  for (int i = 0; i < sz; ++i) {
    img[i] = dis(gen);
  }
  return img;
}

void run_test(int num_rows, int num_cols, const std::vector<double> &input_matrix,
              const std::vector<double> &expected_result, bool expected_validation = true) {
  std::vector<double> output_result(num_rows * num_cols, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<double *>(input_matrix.data())));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_rows));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_cols));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
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

TEST(plekhanov_d_verticalgaus_seq, Matrix3x3_1) {
  plekhanov_d_verticalgaus_seq::run_test(3, 3, {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0},
                                         {0, 0, 0, 0, 1.375, 0, 0, 0, 0});
}

TEST(plekhanov_d_verticalgaus_seq, Matrix3x3_2) {
  plekhanov_d_verticalgaus_seq::run_test(3, 3, {2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
                                         {0, 0, 0, 0, 4.9375, 0, 0, 0, 0});
}

TEST(plekhanov_d_verticalgaus_seq, Matrix4x4_1) {
  plekhanov_d_verticalgaus_seq::run_test(
      4, 4, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
      {0, 0, 0, 0, 0, 6, 7, 0, 0, 10, 11, 0, 0, 0, 0, 0});
}

TEST(plekhanov_d_verticalgaus_seq, Matrix4x4_2) {
  plekhanov_d_verticalgaus_seq::run_test(
      4, 4, {2.0, 1.0, 0.0, 3.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
      {0, 0, 0, 0, 0, 1.5, 2.25, 0, 0, 2.25, 3.25, 0, 0, 0, 0, 0});
}

TEST(plekhanov_d_verticalgaus_seq, Matrix5x5_1) {
  plekhanov_d_verticalgaus_seq::run_test(
      5, 5, {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
             14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0},
      {0, 0, 0, 0, 0, 0, 7, 8, 9, 0, 0, 12, 13, 14, 0, 0, 17, 18, 19, 0, 0, 0, 0, 0, 0});
}

TEST(plekhanov_d_verticalgaus_seq, Matrix5x5_2) {
  plekhanov_d_verticalgaus_seq::run_test(
      5, 5, {5.0,  4.0,  3.0,  2.0,  1.0,  10.0, 9.0,  8.0,  7.0,  6.0,  15.0, 14.0, 13.0,
             12.0, 11.0, 20.0, 19.0, 18.0, 17.0, 16.0, 25.0, 24.0, 23.0, 22.0, 21.0},
      {0, 0, 0, 0, 0, 0, 9, 8, 7, 0, 0, 14, 13, 12, 0, 0, 19, 18, 17, 0, 0, 0, 0, 0, 0});
}
