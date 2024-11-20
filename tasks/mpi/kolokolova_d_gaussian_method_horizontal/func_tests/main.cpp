#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kolokolova_d_gaussian_method_horizontal/include/ops_mpi.hpp"

TEST(kolokolova_d_gaussian_method_horizontal_mpi, Test_Parallel_Gauss1) {
  boost::mpi::communicator world;
  int count_equations = world.size();
  int size_coef_mat = count_equations * count_equations;
  std::vector<int> input_coeff(size_coef_mat, 0);
  std::vector<int> input_y(count_equations, 0);
  std::vector<double> func_res(count_equations, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_y = kolokolova_d_gaussian_method_horizontal_mpi::getRandomVector(count_equations);
    input_coeff = kolokolova_d_gaussian_method_horizontal_mpi::getRandomVector(size_coef_mat);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
    taskDataPar->inputs_count.emplace_back(input_coeff.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
    taskDataPar->inputs_count.emplace_back(input_y.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
    taskDataPar->outputs_count.emplace_back(func_res.size());
  }

  kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_res(count_equations, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
    taskDataSeq->inputs_count.emplace_back(input_coeff.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
    taskDataSeq->inputs_count.emplace_back(input_y.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < count_equations; i++) {
      ASSERT_EQ(func_res[i], reference_res[i]);
    }
  }
}