#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <vector>

#include "mpi/bessonov_e_star_topology/include/ops_mpi.hpp"

TEST(bessonov_e_star_topology_mpi, DataTransmissionTest) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int data_size = 5;
  int input_data[data_size] = {1, 2, 3, 4, 6};

  int traversal_size = 2 * (world.size() - 1) + 1;

  std::vector<int> output_data;
  std::vector<int> traversal_order;

  if (world.rank() == 0) {
    output_data.resize(data_size);
    traversal_order.resize(traversal_size);

    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(input_data));
    taskDataPar->inputs_count.push_back(data_size);

    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(traversal_order.data()));

    taskDataPar->outputs_count.push_back(data_size);
    taskDataPar->outputs_count.push_back(traversal_size);
  }

  bessonov_e_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < data_size; ++i) {
      ASSERT_EQ(output_data[i], input_data[i]);
    }

    std::vector<int> expected_traversal;
    expected_traversal.push_back(0);
    for (int i = 1; i < world.size(); ++i) {
      expected_traversal.push_back(i);
      expected_traversal.push_back(0);
    }
    for (int i = 0; i < traversal_size; ++i) {
      ASSERT_EQ(traversal_order[i], expected_traversal[i]);
    }
  }
}

TEST(bessonov_e_star_topology_mpi, LargeDataTest) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    GTEST_SKIP();
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int data_size = 100000;
  std::vector<int> input_data(data_size, 1);

  int traversal_size = 2 * (world.size() - 1) + 1;

  std::vector<int> output_data;
  std::vector<int> traversal_order;

  if (world.rank() == 0) {
    output_data.resize(data_size);
    traversal_order.resize(traversal_size);

    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskDataPar->inputs_count.push_back(data_size);

    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(traversal_order.data()));

    taskDataPar->outputs_count.push_back(data_size);
    taskDataPar->outputs_count.push_back(traversal_size);
  }

  bessonov_e_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
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
    for (int i = 0; i < traversal_size; ++i) {
      ASSERT_EQ(traversal_order[i], expected_traversal[i]);
    }
  }
}

TEST(bessonov_e_star_topology_mpi, RandomDataTest) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    GTEST_SKIP();
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int data_size = 100;
  std::vector<int> input_data(data_size);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(-1000, 1000);
  for (int& val : input_data) {
    val = dist(gen);
  }

  int traversal_size = 2 * (world.size() - 1) + 1;

  std::vector<int> output_data;
  std::vector<int> traversal_order;

  if (world.rank() == 0) {
    output_data.resize(data_size);
    traversal_order.resize(traversal_size);

    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskDataPar->inputs_count.push_back(data_size);

    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(traversal_order.data()));

    taskDataPar->outputs_count.push_back(data_size);
    taskDataPar->outputs_count.push_back(traversal_size);
  }

  bessonov_e_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < data_size; ++i) {
      ASSERT_EQ(output_data[i], input_data[i]);
    }

    std::vector<int> expected_traversal;
    expected_traversal.push_back(0);
    for (int i = 1; i < world.size(); ++i) {
      expected_traversal.push_back(i);
      expected_traversal.push_back(0);
    }
    for (int i = 0; i < traversal_size; ++i) {
      ASSERT_EQ(traversal_order[i], expected_traversal[i]);
    }
  }
}

TEST(bessonov_e_star_topology_mpi, DataTest_2_power_10) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    GTEST_SKIP();
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int data_size = 1024;
  std::vector<int> input_data(data_size, 1);

  int traversal_size = 2 * (world.size() - 1) + 1;

  std::vector<int> output_data;
  std::vector<int> traversal_order;

  if (world.rank() == 0) {
    output_data.resize(data_size);
    traversal_order.resize(traversal_size);

    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskDataPar->inputs_count.push_back(data_size);

    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(traversal_order.data()));

    taskDataPar->outputs_count.push_back(data_size);
    taskDataPar->outputs_count.push_back(traversal_size);
  }

  bessonov_e_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
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
    for (int i = 0; i < traversal_size; ++i) {
      ASSERT_EQ(traversal_order[i], expected_traversal[i]);
    }
  }
}

TEST(bessonov_e_star_topology_mpi, DataTest_2_power_13) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    GTEST_SKIP();
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int data_size = 8192;
  std::vector<int> input_data(data_size, 1);

  int traversal_size = 2 * (world.size() - 1) + 1;

  std::vector<int> output_data;
  std::vector<int> traversal_order;

  if (world.rank() == 0) {
    output_data.resize(data_size);
    traversal_order.resize(traversal_size);

    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskDataPar->inputs_count.push_back(data_size);

    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(traversal_order.data()));

    taskDataPar->outputs_count.push_back(data_size);
    taskDataPar->outputs_count.push_back(traversal_size);
  }

  bessonov_e_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
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
    for (int i = 0; i < traversal_size; ++i) {
      ASSERT_EQ(traversal_order[i], expected_traversal[i]);
    }
  }
}

TEST(bessonov_e_star_topology_mpi, DataTest_3_power_8) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    GTEST_SKIP();
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int data_size = 6561;
  std::vector<int> input_data(data_size, 1);

  int traversal_size = 2 * (world.size() - 1) + 1;

  std::vector<int> output_data;
  std::vector<int> traversal_order;

  if (world.rank() == 0) {
    output_data.resize(data_size);
    traversal_order.resize(traversal_size);

    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskDataPar->inputs_count.push_back(data_size);

    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(traversal_order.data()));

    taskDataPar->outputs_count.push_back(data_size);
    taskDataPar->outputs_count.push_back(traversal_size);
  }

  bessonov_e_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
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
    for (int i = 0; i < traversal_size; ++i) {
      ASSERT_EQ(traversal_order[i], expected_traversal[i]);
    }
  }
}

TEST(bessonov_e_star_topology_mpi, DataTest_3_power_10) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    GTEST_SKIP();
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int data_size = 59049;
  std::vector<int> input_data(data_size, 1);

  int traversal_size = 2 * (world.size() - 1) + 1;

  std::vector<int> output_data;
  std::vector<int> traversal_order;

  if (world.rank() == 0) {
    output_data.resize(data_size);
    traversal_order.resize(traversal_size);

    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskDataPar->inputs_count.push_back(data_size);

    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(traversal_order.data()));

    taskDataPar->outputs_count.push_back(data_size);
    taskDataPar->outputs_count.push_back(traversal_size);
  }

  bessonov_e_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
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
    for (int i = 0; i < traversal_size; ++i) {
      ASSERT_EQ(traversal_order[i], expected_traversal[i]);
    }
  }
}

TEST(bessonov_e_star_topology_mpi, DataTest_prime_values_2003) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    GTEST_SKIP();
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int data_size = 2003;
  std::vector<int> input_data(data_size, 1);

  int traversal_size = 2 * (world.size() - 1) + 1;

  std::vector<int> output_data;
  std::vector<int> traversal_order;

  if (world.rank() == 0) {
    output_data.resize(data_size);
    traversal_order.resize(traversal_size);

    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskDataPar->inputs_count.push_back(data_size);

    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(traversal_order.data()));

    taskDataPar->outputs_count.push_back(data_size);
    taskDataPar->outputs_count.push_back(traversal_size);
  }

  bessonov_e_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
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
    for (int i = 0; i < traversal_size; ++i) {
      ASSERT_EQ(traversal_order[i], expected_traversal[i]);
    }
  }
}

TEST(bessonov_e_star_topology_mpi, DataTest_prime_values_4993) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    GTEST_SKIP();
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int data_size = 4993;
  std::vector<int> input_data(data_size, 1);

  int traversal_size = 2 * (world.size() - 1) + 1;

  std::vector<int> output_data;
  std::vector<int> traversal_order;

  if (world.rank() == 0) {
    output_data.resize(data_size);
    traversal_order.resize(traversal_size);

    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskDataPar->inputs_count.push_back(data_size);

    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(traversal_order.data()));

    taskDataPar->outputs_count.push_back(data_size);
    taskDataPar->outputs_count.push_back(traversal_size);
  }

  bessonov_e_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
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
    for (int i = 0; i < traversal_size; ++i) {
      ASSERT_EQ(traversal_order[i], expected_traversal[i]);
    }
  }
}

TEST(bessonov_e_star_topology_mpi, ValidationTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  bessonov_e_star_topology_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}