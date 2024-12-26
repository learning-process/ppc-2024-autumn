#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <climits>
#include <vector>

#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

namespace malyshev_v_conjugate_gradient_method_mpi {

void test_task(const std::vector<std::vector<double>> &A, const std::vector<double> &b, const std::vector<double> &ref,
               double eps) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel taskMPI(taskDataMPI, A, b);
  malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential taskSeq(taskDataSeq, A, b);

  std::vector<double> resSeq(b.size());
  std::vector<double> resMPI(b.size());

  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resMPI));
    taskDataMPI->inputs_count.emplace_back(1);
    taskDataMPI->inputs_count.emplace_back(1);
    taskDataMPI->outputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resSeq));
    taskDataSeq->inputs = taskDataMPI->inputs;
    taskDataSeq->inputs_count = taskDataMPI->inputs_count;
    taskDataSeq->outputs_count = taskDataMPI->outputs_count;
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (size_t i = 0; i < ref.size(); ++i) {
      ASSERT_NEAR(resMPI[i], resSeq[i], eps);
      ASSERT_NEAR(resMPI[i], ref[i], eps);
    }
  }
}

}  // namespace malyshev_v_conjugate_gradient_method_mpi

TEST(malyshev_v_conjugate_gradient_method_mpi, SimpleTest) {
  std::vector<std::vector<double>> A = {{4, 1}, {1, 3}};
  std::vector<double> b = {1, 2};
  std::vector<double> ref = {0.0909, 0.6364};
  double eps = 1e-4;

  test_task(A, b, ref, eps);
}

TEST(malyshev_v_conjugate_gradient_method_mpi, MediumTest) {
  std::vector<std::vector<double>> A = {{5, 2, 0}, {2, 5, 2}, {0, 2, 5}};
  std::vector<double> b = {1, 2, 3};
  std::vector<double> ref = {0.0769, 0.2308, 0.5385};
  double eps = 1e-4;

  test_task(A, b, ref, eps);
}

TEST(malyshev_v_conjugate_gradient_method_mpi, LargeTest) {
  std::vector<std::vector<double>> A = {{8, 3, 0, 0}, {3, 9, 4, 0}, {0, 4, 10, 5}, {0, 0, 5, 12}};
  std::vector<double> b = {1, 2, 3, 4};
  std::vector<double> ref = {0.0806, 0.1613, 0.1935, 0.2903};
  double eps = 1e-4;

  test_task(A, b, ref, eps);
}