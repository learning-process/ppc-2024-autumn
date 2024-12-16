#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/prokhorov_n_producer_customer/include/ops_mpi.hpp"

TEST(prokhorov_n_producer_customer_mpi, Test_Sequence_Numbers_Processes) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sum;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_producers = 10;

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      global_vec.push_back(i + 1);
    }
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  prokhorov_n_producer_customer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      ASSERT_EQ(global_vec[i], global_sum[i]);
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Doubled_Numbers_Processes) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sum;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_producers = 10;

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      global_vec.push_back((i + 1) * 2);
    }
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  prokhorov_n_producer_customer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      ASSERT_EQ(global_vec[i], global_sum[i]);
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Reverse_Sequence_Numbers_Processes) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sum;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_producers = 10;

  if (world.rank() == 0) {
    for (int i = num_producers; i > 0; i--) {
      global_vec.push_back(i);
    }
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  prokhorov_n_producer_customer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      ASSERT_EQ(global_vec[i], global_sum[i]);
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Multiples_Of_Five_Processes) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sum;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_producers = 10;

  if (world.rank() == 0) {
    for (int i = 1; i <= num_producers; i++) {
      global_vec.push_back(i * 5);
    }
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  prokhorov_n_producer_customer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      ASSERT_EQ(global_vec[i], global_sum[i]);
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Squares_Of_Numbers_Processes) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sum;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_producers = 10;

  if (world.rank() == 0) {
    for (int i = 1; i <= num_producers; i++) {
      global_vec.push_back(i * i);
    }
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  prokhorov_n_producer_customer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      ASSERT_EQ(global_vec[i], global_sum[i]);
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Odd_Numbers_Processes) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sum;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_producers = 10;

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      global_vec.push_back(2 * i + 1);
    }
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  prokhorov_n_producer_customer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      ASSERT_EQ(global_vec[i], global_sum[i]);
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Negative_Numbers_Processes) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sum;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_producers = 10;

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      global_vec.push_back(-1 * (i + 1));
    }
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  prokhorov_n_producer_customer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      ASSERT_EQ(global_vec[i], global_sum[i]);
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Even_Numbers_Processes) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sum;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_producers = 10;

  if (world.rank() == 0) {
    for (int i = 1; i <= num_producers; i++) {
      global_vec.push_back(i * 2);
    }
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  prokhorov_n_producer_customer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      ASSERT_EQ(global_vec[i], global_sum[i]);
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Cubes_Of_Numbers_Processes) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sum;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_producers = 10;

  if (world.rank() == 0) {
    for (int i = 1; i <= num_producers; i++) {
      global_vec.push_back(i * i * i);
    }
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  prokhorov_n_producer_customer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      ASSERT_EQ(global_vec[i], global_sum[i]);
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Multiples_Of_Three_Processes) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sum;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_producers = 10;

  if (world.rank() == 0) {
    for (int i = 1; i <= num_producers; i++) {
      global_vec.push_back(i * 3);
    }
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  prokhorov_n_producer_customer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      ASSERT_EQ(global_vec[i], global_sum[i]);
    }
  }
}
