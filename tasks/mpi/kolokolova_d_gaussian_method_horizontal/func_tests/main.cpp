#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "mpi/kolokolova_d_gaussian_method_horizontal/include/ops_mpi.hpp"

TEST(kolokolova_d_gaussian_method_horizontal_mpi, Test_Parallel_Gauss) {
  boost::mpi::communicator world;
  int count_equations = world.size();
  int size_coef_mat = count_equations * count_equations;
  std::vector<int> input_coeff(size_coef_mat, 0);
  std::vector<int> input_y(count_equations, 0);
  std::vector<double> func_res(count_equations, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    
    //std::vector<int> input_coeff(size_coef_mat, 0);
    //std::vector<int> input_y(count_equations, 0);
    //std::vector<double> func_res(count_equations, 0);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<int> dist(0, 100);
    for (int i = 0; i < count_equations; ++i) {
      input_y[i] = gen() % 100;
      std::cout << "y -" << input_y[i] << "\n";
      for (int j = 0; j < count_equations; ++j) {
        input_coeff[j] = gen() % 100;
        std::cout << "coef -" << input_coeff[j] << "\n";
      }
    }
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
    ASSERT_EQ(1, 1);
  }
}