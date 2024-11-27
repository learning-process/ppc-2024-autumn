#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <vector>

#include "mpi/bessonov_e_star_topology/include/ops_mpi.hpp"

TEST(bessonov_e_star_topology_mpi, DataTransmissionTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int data_size = 5;
  int input_data[data_size] = {1, 2, 3, 4, 6};

  // Calculating the traversal_order size_
  int traversal_size = 2 * (world.size() - 1) + 1;

  int* output_data = nullptr;
  int* traversal_order = nullptr;

  if (world.rank() == 0) {
    output_data = new int[data_size];
    traversal_order = new int[traversal_size];

    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(input_data));
    taskDataPar->inputs_count.push_back(data_size);

    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(output_data));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(traversal_order));

    taskDataPar->outputs_count.push_back(data_size);
    taskDataPar->outputs_count.push_back(traversal_size);
  }

  bessonov_e_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  bool isValid = testMpiTaskParallel.validation();
  if (!isValid) {
    GTEST_SKIP();
  }
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Checking that the data has not changed
    for (int i = 0; i < data_size; ++i) {
      ASSERT_EQ(output_data[i], input_data[i]);
    }

    // Checking the traversal order
    std::vector<int> expected_traversal;
    expected_traversal.push_back(0);
    for (int i = 1; i < world.size(); ++i) {
      expected_traversal.push_back(i);
      expected_traversal.push_back(0);
    }
    if (world.rank() == 0) {
      for (int i = 0; i < traversal_size; ++i) {
        ASSERT_EQ(traversal_order[i], expected_traversal[i]);
      }
    }

    delete[] output_data;
    delete[] traversal_order;
  }
}

TEST(bessonov_e_star_topology_mpi, LargeDataTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int data_size = 100000;
  std::vector<int> input_data(data_size, 1);

  int traversal_size = 2 * (world.size() - 1) + 1;

  int* output_data = nullptr;
  int* traversal_order = nullptr;

  if (world.rank() == 0) {
    output_data = new int[data_size];
    traversal_order = new int[traversal_size];

    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskDataPar->inputs_count.push_back(data_size);

    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(output_data));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(traversal_order));

    taskDataPar->outputs_count.push_back(data_size);
    taskDataPar->outputs_count.push_back(traversal_size);
  }

  bessonov_e_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  bool isValid = testMpiTaskParallel.validation();
  if (!isValid) {
    GTEST_SKIP();
  }
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < data_size; ++i) {
      ASSERT_EQ(output_data[i], 1);
    }

    std::vector<int> expected_traversal;
    expected_traversal.push_back(0);
    for (int i = 1; i < world.size(); ++i) {
      expected_traversal.push_back(i);
      expected_traversal.push_back(0);
    }
    if (world.rank() == 0) {
      for (int i = 0; i < traversal_size; ++i) {
        ASSERT_EQ(traversal_order[i], expected_traversal[i]);
      }
    }

    delete[] output_data;
    delete[] traversal_order;
  }
}

TEST(bessonov_e_star_topology_mpi, ValidationTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  // not provide input data to cause an error in validation()

  bessonov_e_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}