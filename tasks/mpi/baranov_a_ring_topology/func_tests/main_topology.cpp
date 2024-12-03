#include <gtest/gtest.h>

#include "mpi/baranov_a_ring_topology/include/header_topology.hpp"
template <typename tp>
typename std::enable_if<std::is_arithmetic<tp>::value>::type get_rnd_vec(std::vector<tp> &vec) {
  std::random_device rd;
  std::default_random_engine reng(rd());

  if constexpr (std::is_integral<tp>::value) {
    // Для целых чисел
    std::uniform_int_distribution<tp> dist(0, vec.size());
    std::generate(vec.begin(), vec.end(), [&dist, &reng] { return dist(reng); });
  } else if constexpr (std::is_floating_point<tp>::value) {
    // Для вещественных чисел
    std::uniform_real_distribution<tp> dist(0.0, vec.size());
    std::generate(vec.begin(), vec.end(), [&dist, &reng] { return dist(reng); });
  }
}

TEST(baranov_a_ring_topology, Test_ring_0_int) {
  const int N = 0;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out_vec(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<int> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(true, std::equal(arr.begin(), arr.end(), out_vec.begin(), out_vec.end()));
  }
}

TEST(baranov_a_ring_topology, Test_ring_10_int) {
  const int N = 10;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out_vec(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<int> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(true, std::equal(arr.begin(), arr.end(), out_vec.begin(), out_vec.end()));
  }
}

TEST(baranov_a_ring_topology, Test_ring_1000_int) {
  const int N = 1000;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out_vec(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<int> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(true, std::equal(arr.begin(), arr.end(), out_vec.begin(), out_vec.end()));
  }
}

TEST(baranov_a_ring_topology, Test_ring_10000_int) {
  const int N = 10000;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out_vec(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<int> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(true, std::equal(arr.begin(), arr.end(), out_vec.begin(), out_vec.end()));
  }
}

TEST(baranov_a_ring_topology, Test_ring_100_double) {
  const int N = 100;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<double> out_vec(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<double> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(true, std::equal(arr.begin(), arr.end(), out_vec.begin(), out_vec.end()));
  }
}

TEST(baranov_a_ring_topology, Test_ring_1000_double) {
  const int N = 1000;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<double> out(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<double> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(arr, out);
  }
}
TEST(baranov_a_ring_topology, Test_ring_100000_double) {
  const int N = 100000;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<double> out(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<double> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(arr, out);
  }
}
TEST(baranov_a_ring_topology, Test_ring_100_uint) {
  const int N = 100;
  // Create data
  boost::mpi::communicator world;
  std::vector<unsigned> arr(N);
  std::vector<unsigned> out(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<unsigned> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(arr, out);
  }
}
TEST(baranov_a_ring_topology, Test_ring_10000_uint) {
  const int N = 10000;
  // Create data
  boost::mpi::communicator world;
  std::vector<unsigned> arr(N);
  std::vector<unsigned> out(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<unsigned> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(arr, out);
  }
}

TEST(baranov_a_ring_topology, Test_ring_prime_number_uint) {
  const int N = 563;
  // Create data
  boost::mpi::communicator world;
  std::vector<unsigned> arr(N);
  std::vector<unsigned> out(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<unsigned> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(arr, out);
  }
}

TEST(baranov_a_ring_topology, Test_ring_prime_number_double) {
  const int N = 563;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<double> out(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<double> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(arr, out);
  }
}

TEST(baranov_a_ring_topology, Test_ring_1024_int_two_to_the_power_of_10) {
  const int N = 1024;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<int> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(arr, out);
  }
}

TEST(baranov_a_ring_topology, Test_ring_32768_int_two_to_the_power_of_17) {
  const int N = 32768;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<int> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(arr, out);
  }
}

TEST(baranov_a_ring_topology, Test_ring_512_double_two_to_the_power_of_9) {
  const int N = 512;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<double> out(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<double> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(arr, out);
  }
}

TEST(baranov_a_ring_topology, Test_ring_4096_double_two_to_the_power_of_12) {
  const int N = 4096;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<double> out(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<double> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(arr, out);
  }
}

TEST(baranov_a_ring_topology, Test_ring_32768_double_two_to_the_power_of_17) {
  const int N = 32768;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<double> out(N);
  std::vector<int> out_poll;
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    out_poll = std::vector<int>(world.size());
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_poll.data()));
    data->outputs_count.emplace_back(2);
  }
  baranov_a_ring_topology_mpi::ring_topology<double> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_poll[i]);
      }
    }
    ASSERT_EQ(arr, out);
  }
}