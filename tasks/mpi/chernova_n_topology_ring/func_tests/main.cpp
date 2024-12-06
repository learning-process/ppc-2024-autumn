#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/chernova_n_topology_ring/include/ops_mpi.hpp"

std::vector<char> generateData(int k) {
  const std::string words[] = {"one", "two", "three"};

  std::string result;
  int j = words->size();

  for (int i = 0; i < k; ++i) {
    result += words[rand() % (j)];
    if (i < k - 1) {
      result += ' ';
    }
  }

  return std::vector<char>(result.begin(), result.end());
}

TEST(chernova_n_topology_ring_mpi, Test_empty_string) {
  boost::mpi::communicator world;
  std::vector<char> in = {};
  const int N = in.size();
  std::vector<char> out_vec(N);
  std::vector<int> out_process;
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size()+1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_process.data()));
    taskDataParallel->outputs_count.emplace_back(2);
  }
  chernova_n_topology_ring_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_process[i]);
      }
    }
    ASSERT_EQ(true, std::equal(in.begin(), in.end(), out_vec.begin(), out_vec.end()));
  }
}
TEST(chernova_n_topology_ring_mpi, Test_ten_symbols) {
  boost::mpi::communicator world;
  std::vector<char> in = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'};
  const int N = in.size();
  std::vector<char> out_vec(N);
  std::vector<int> out_process;
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size()+1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_process.data()));
    taskDataParallel->outputs_count.emplace_back(2);
  }
  chernova_n_topology_ring_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_process[i]);
      }
    }
    ASSERT_EQ(true, std::equal(in.begin(), in.end(), out_vec.begin(), out_vec.end()));
  }
}

TEST(chernova_n_topology_ring_mpi, Test_five_words) {
  boost::mpi::communicator world;
  std::vector<char> in = generateData(5);
  const int N = in.size();
  std::vector<char> out_vec(N);
  std::vector<int> out_process;
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size()+1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_process.data()));
    taskDataParallel->outputs_count.emplace_back(2);
  }
  chernova_n_topology_ring_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_process[i]);
      }
    }
    ASSERT_EQ(true, std::equal(in.begin(), in.end(), out_vec.begin(), out_vec.end()));
  }
}

TEST(chernova_n_topology_ring_mpi, Test_ten_words) {
  boost::mpi::communicator world;
  std::vector<char> in = generateData(10);
  const int N = in.size();
  std::vector<char> out_vec(N);
  std::vector<int> out_process;
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size()+1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_process.data()));
    taskDataParallel->outputs_count.emplace_back(2);
  }
  chernova_n_topology_ring_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_process[i]);
      }
    }
    ASSERT_EQ(true, std::equal(in.begin(), in.end(), out_vec.begin(), out_vec.end()));
  }
}

TEST(chernova_n_topology_ring_mpi, Test_twenty_words) {
  boost::mpi::communicator world;
  std::vector<char> in = generateData(20);
  const int N = in.size();
  std::vector<char> out_vec(N);
  std::vector<int> out_process;
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size()+1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_process.data()));
    taskDataParallel->outputs_count.emplace_back(2);
  }
  chernova_n_topology_ring_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_process[i]);
      }
    }
    ASSERT_EQ(true, std::equal(in.begin(), in.end(), out_vec.begin(), out_vec.end()));
  }
}

TEST(chernova_n_topology_ring_mpi, Test_thirty_words) {
  boost::mpi::communicator world;
  std::vector<char> in = generateData(30);
  const int N = in.size();
  std::vector<char> out_vec(N);
  std::vector<int> out_process;
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size()+1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_process.data()));
    taskDataParallel->outputs_count.emplace_back(2);
  }
  chernova_n_topology_ring_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_process[i]);
      }
    }
    ASSERT_EQ(true, std::equal(in.begin(), in.end(), out_vec.begin(), out_vec.end()));
  }
}