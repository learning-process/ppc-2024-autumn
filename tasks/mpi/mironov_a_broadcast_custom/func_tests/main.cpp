#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/mironov_a_broadcast_custom/include/ops_mpi.hpp"

namespace mironov_a_broadcast_custom_mpi {

std::vector<int> get_random_vector(int sz, bool is_powers = false) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);

  if (is_powers) {
    const int mod = 5;
    for (int i = 0; i < sz; i++) {
      vec[i] = gen() % mod;
    }
  } else {
    const int mod = 100;
    for (int i = 0; i < sz; i++) {
      vec[i] = gen() % mod - 50;
    }
  }

  return vec;
}
}  // namespace mironov_a_broadcast_custom_mpi

TEST(mironov_a_broadcast_custom_mpi, Test_broadcast_1) {
  boost::mpi::communicator world;
  // Create data
  const std::vector<int> golds = {3, 14, 39};
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    global_input = std::vector<int>({1, 2, 3});
    global_powers = std::vector<int>({1, 2, 3});

    global_res.resize(3, 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  mironov_a_broadcast_custom_mpi::ComponentSumPowerCustomImpl testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_res, golds);
  }
}

TEST(mironov_a_broadcast_custom_mpi, Test_broadcast_2) {
  boost::mpi::communicator world;
  // Create data
  const std::vector<int> golds = {10, 20, 30, 10, 20, 30};
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    global_input = std::vector<int>({1, 2, 3, 1, 2, 3});
    global_powers = std::vector<int>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

    global_res.resize(global_input.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  mironov_a_broadcast_custom_mpi::ComponentSumPowerCustomImpl testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_res, golds);
  }
}

TEST(mironov_a_broadcast_custom_mpi, Test_broadcast_3) {
  boost::mpi::communicator world;
  // Create data
  const std::vector<int> golds = {static_cast<int>(pow(2, 20))};
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    global_input = std::vector<int>({2});
    global_powers = std::vector<int>({20});

    global_res.resize(global_input.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  mironov_a_broadcast_custom_mpi::ComponentSumPowerCustomImpl testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_res, golds);
  }
}

TEST(mironov_a_broadcast_custom_mpi, Test_broadcast_4) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> golds;
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    global_input = std::vector<int>(50);
    global_powers = std::vector<int>(20);
    golds = std::vector<int>(50);

    for (int i = 0; i < 10; ++i) {
      global_powers[2 * i] = i;
      global_powers[2 * i + 1] = i;
    }
    for (int i = 0; i < 50; ++i) {
      global_input[i] = (i + 1) / 10;
      for (auto power : global_powers) {
        golds[i] += static_cast<int>(pow(global_input[i], power));
      }
    }

    global_res.resize(global_input.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  mironov_a_broadcast_custom_mpi::ComponentSumPowerCustomImpl testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_res, golds);
  }
}

TEST(mironov_a_broadcast_custom_mpi, Test_broadcast_random_1) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> golds;
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    global_input = mironov_a_broadcast_custom_mpi::get_random_vector(50);
    global_powers = mironov_a_broadcast_custom_mpi::get_random_vector(20, true);
    golds = std::vector<int>(50);

    for (int i = 0; i < 50; ++i) {
      for (auto power : global_powers) {
        golds[i] += static_cast<int>(pow(global_input[i], power));
      }
    }

    global_res.resize(global_input.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  mironov_a_broadcast_custom_mpi::ComponentSumPowerCustomImpl testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_res, golds);
  }
}

TEST(mironov_a_broadcast_custom_mpi, Test_broadcast_random_2) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> golds;
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    global_input = mironov_a_broadcast_custom_mpi::get_random_vector(33);
    global_powers = mironov_a_broadcast_custom_mpi::get_random_vector(10, true);
    golds = std::vector<int>(33);

    for (int i = 0; i < 33; ++i) {
      for (auto power : global_powers) {
        golds[i] += static_cast<int>(pow(global_input[i], power));
      }
    }

    global_res.resize(global_input.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  mironov_a_broadcast_custom_mpi::ComponentSumPowerCustomImpl testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_res, golds);
  }
}

TEST(mironov_a_broadcast_custom_mpi, Test_broadcast_random_64) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> golds;
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    global_input = mironov_a_broadcast_custom_mpi::get_random_vector(64);
    global_powers = mironov_a_broadcast_custom_mpi::get_random_vector(8, true);
    golds = std::vector<int>(64);

    for (int i = 0; i < 64; ++i) {
      for (auto power : global_powers) {
        golds[i] += static_cast<int>(pow(global_input[i], power));
      }
    }

    global_res.resize(global_input.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  mironov_a_broadcast_custom_mpi::ComponentSumPowerCustomImpl testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_res, golds);
  }
}

TEST(mironov_a_broadcast_custom_mpi, Test_broadcast_random_128) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> golds;
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    global_input = mironov_a_broadcast_custom_mpi::get_random_vector(128);
    global_powers = mironov_a_broadcast_custom_mpi::get_random_vector(8, true);
    golds = std::vector<int>(128);

    for (int i = 0; i < 128; ++i) {
      for (auto power : global_powers) {
        golds[i] += static_cast<int>(pow(global_input[i], power));
      }
    }

    global_res.resize(global_input.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  mironov_a_broadcast_custom_mpi::ComponentSumPowerCustomImpl testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_res, golds);
  }
}

TEST(mironov_a_broadcast_custom_mpi, Test_wrong_input_1) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> golds;
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    global_input = std::vector<int>();
    global_powers = std::vector<int>(20);
    golds = std::vector<int>();

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  mironov_a_broadcast_custom_mpi::ComponentSumPowerCustomImpl testMpiTaskParallel(taskDataGlob);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(mironov_a_broadcast_custom_mpi, Test_wrong_input_2) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> golds;
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    global_input = std::vector<int>(10, 1);
    global_powers = std::vector<int>();
    golds = std::vector<int>(10, 1);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  mironov_a_broadcast_custom_mpi::ComponentSumPowerCustomImpl testMpiTaskParallel(taskDataGlob);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(mironov_a_broadcast_custom_mpi, Test_broadcast_with_diff_root_1) {
  boost::mpi::communicator world;
  if (world.size() < 3) {
    return;
  }
  // Create data
  std::vector<int> vector_for_broadcast(10);
  std::vector<int> golds = std::vector<int>({1, 2, 3, 4, 5, 6, 7, 10, 9, 8});
  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  // the third process is root
  if (world.rank() == 2) {
    vector_for_broadcast = std::vector<int>({1, 2, 3, 4, 5, 6, 7, 10, 9, 8});
  }

  // broadcast vector
  mironov_a_broadcast_custom_mpi::ComponentSumPowerCustomImpl::broadcastImpl(world, vector_for_broadcast.data(), 10, 2);

  // check data in every process
  ASSERT_EQ(vector_for_broadcast, golds);
}

TEST(mironov_a_broadcast_custom_mpi, Test_broadcast_with_diff_root_2) {
  boost::mpi::communicator world;
  if (world.size() < 3) {
    return;
  }
  // Create data
  std::vector<int> vector_for_broadcast(10);
  std::vector<int> golds = std::vector<int>({1, 2, 3, 4, 5, 6, 7, 10, 9, 8});
  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  // the second process is root
  if (world.rank() == 1) {
    vector_for_broadcast = std::vector<int>({1, 2, 3, 4, 5, 6, 7, 10, 9, 8});
  }

  // broadcast vector
  mironov_a_broadcast_custom_mpi::ComponentSumPowerCustomImpl::broadcastImpl(world, vector_for_broadcast.data(), 10, 1);

  // check data in every process
  ASSERT_EQ(vector_for_broadcast, golds);
}

TEST(mironov_a_broadcast_custom_mpi, Test_broadcast_with_even_and_odd_processes) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  // here using 2 roots: 3 and 4 processes
  if (world.size() < 4) {
    return;
  }

  // initialize data and golds
  std::vector<int> vector_for_broadcast(10);
  std::vector<int> golds_even = {1, 2, 3, 4, 5, 6, 7, 10, 9, 8};
  std::vector<int> golds_odd = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();

  int rank = world.rank();

  // create another communicator
  boost::mpi::communicator local_comm = world.split(rank % 2);
  if (rank == 2 || rank == 3) {
    vector_for_broadcast = golds_even;
  }

  // share data
  mironov_a_broadcast_custom_mpi::ComponentSumPowerCustomImpl::broadcastImpl(local_comm, vector_for_broadcast.data(),
                                                                             vector_for_broadcast.size(), 1);

  // reduce result (need to not to do ASSERT_EQ many times)
  int res = 1;
  if (rank % 2 == 1 && vector_for_broadcast != golds_odd) {
    rank = 0;
  } else if (rank % 2 == 0 && vector_for_broadcast != golds_even) {
    rank = 0;
  }
  boost::mpi::reduce(world, res, boost::mpi::minimum<int>(), 2);
  if (rank == 2) {
    ASSERT_EQ(res, 1);
  }
}
