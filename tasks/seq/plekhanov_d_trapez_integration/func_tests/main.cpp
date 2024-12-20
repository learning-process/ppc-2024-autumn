#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "seq/plekhanov_d_trapez_integration/include/ops_seq.hpp"

namespace plekhanov_d_trapez_integration_seq {

void run_test(double a, double b, double epsilon, double expected_result, bool expected_validation = true) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(1);

  double output = 0.0;
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  taskDataSeq->outputs_count.emplace_back(1);

  plekhanov_d_trapez_integration_seq::TestTaskSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), expected_validation);

  if (expected_validation) {
    taskSequential.pre_processing();
    taskSequential.run();
    taskSequential.post_processing();

    ASSERT_NEAR(output, expected_result, epsilon);
  }
}

}  // namespace plekhanov_d_trapez_integration_seq

TEST(plekhanov_d_trapez_integration_seq, test1) { plekhanov_d_trapez_integration_seq::run_test(1.45, 0, 0.01, -1.016); }

TEST(plekhanov_d_trapez_integration_seq, test2) {
  plekhanov_d_trapez_integration_seq::run_test(0.0, 1.45, 1e-2, 1.016);
}

TEST(plekhanov_d_trapez_integration_seq, test3) {
  plekhanov_d_trapez_integration_seq::run_test(-1.45, 1.45, 0.01, 2.03);
}

TEST(plekhanov_d_trapez_integration_seq, test4) { plekhanov_d_trapez_integration_seq::run_test(1.45, 0, 0.01, -1.016); }

TEST(plekhanov_d_trapez_integration_seq, test5) {
  plekhanov_d_trapez_integration_seq::run_test(0, 100.0, 0.01, 333333.333510);
}

TEST(plekhanov_d_trapez_integration_seq, test6) {
  plekhanov_d_trapez_integration_seq::run_test(-10.0, 65.0, 0.01, 91875.001);
}

TEST(plekhanov_d_trapez_integration_seq, test7) {
  plekhanov_d_trapez_integration_seq::run_test(-10.0, 10.0, 0.01, 666.66666);
}