#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/dormidontov_e_circle_topology/include/dect_ops_mpi.hpp"
namespace dormidontov_e_circle_topology_mpi {
inline std::vector<int> GenVector(int size, int start = -1000, int finish = 1000) {
  std::vector<int> temp(size);
  for (int i = 0; i < size; i++) {
    temp[i] = (rand() % (finish - start + 1)) + start;
  }
  return temp;
}
inline std::vector<int> GenMarksAns(int size) {
  std::vector<int> temp(size);
  for (int i = 0; i < size; i++) {
    temp[i] = i;
  }
  temp.push_back(0);
  return temp;
}
}  // namespace dormidontov_e_circle_topology_mpi

TEST(dormidontov_e_circle_topology_mpi, Test_1) {
  int size = 1;
  boost::mpi::communicator world;
  std::vector<int> mas;
  mas = dormidontov_e_circle_topology_mpi::GenVector(size);
  std::vector<int> ans(size);
  std::vector<int> marks(world.size() + 1);
  std::vector<int> marks_of_ans = dormidontov_e_circle_topology_mpi::GenMarksAns(world.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(marks.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.rank() + 1);
  }
  dormidontov_e_circle_topology_mpi::topology MPITask(taskDataPar);
  ASSERT_EQ(MPITask.validation(), true);
  MPITask.pre_processing();
  MPITask.run();
  MPITask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(mas, ans);
    ASSERT_EQ(marks, marks_of_ans);
  }
}

TEST(dormidontov_e_circle_topology_mpi, Test_10) {
  int size = 10;
  boost::mpi::communicator world;
  std::vector<int> mas;
  mas = dormidontov_e_circle_topology_mpi::GenVector(size);
  std::vector<int> ans(size);
  std::vector<int> marks(world.size() + 1);
  std::vector<int> marks_of_ans = dormidontov_e_circle_topology_mpi::GenMarksAns(world.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(marks.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.rank() + 1);
  }
  dormidontov_e_circle_topology_mpi::topology MPITask(taskDataPar);
  ASSERT_EQ(MPITask.validation(), true);
  MPITask.pre_processing();
  MPITask.run();
  MPITask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(mas, ans);
    ASSERT_EQ(marks, marks_of_ans);
  }
}

TEST(dormidontov_e_circle_topology_mpi, Test_100) {
  int size = 100;
  boost::mpi::communicator world;
  std::vector<int> mas;
  mas = dormidontov_e_circle_topology_mpi::GenVector(size);
  std::vector<int> ans(size);
  std::vector<int> marks(world.size() + 1);
  std::vector<int> marks_of_ans = dormidontov_e_circle_topology_mpi::GenMarksAns(world.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(marks.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.rank() + 1);
  }
  dormidontov_e_circle_topology_mpi::topology MPITask(taskDataPar);
  ASSERT_EQ(MPITask.validation(), true);
  MPITask.pre_processing();
  MPITask.run();
  MPITask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(mas, ans);
    ASSERT_EQ(marks, marks_of_ans);
  }
}

TEST(dormidontov_e_circle_topology_mpi, Test_1000) {
  int size = 1000;
  boost::mpi::communicator world;
  std::vector<int> mas;
  mas = dormidontov_e_circle_topology_mpi::GenVector(size);
  std::vector<int> ans(size);
  std::vector<int> marks(world.size() + 1);
  std::vector<int> marks_of_ans = dormidontov_e_circle_topology_mpi::GenMarksAns(world.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(marks.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.rank() + 1);
  }
  dormidontov_e_circle_topology_mpi::topology MPITask(taskDataPar);
  ASSERT_EQ(MPITask.validation(), true);
  MPITask.pre_processing();
  MPITask.run();
  MPITask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(mas, ans);
    ASSERT_EQ(marks, marks_of_ans);
  }
}

TEST(dormidontov_e_circle_topology_mpi, Test_if_size_zero) {
  int size = 0;
  boost::mpi::communicator world;
  std::vector<int> mas;
  mas = dormidontov_e_circle_topology_mpi::GenVector(size);
  std::vector<int> ans(size);
  std::vector<int> marks(world.size() + 1);
  std::vector<int> marks_of_ans = dormidontov_e_circle_topology_mpi::GenMarksAns(world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(marks.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.rank() + 1);
    dormidontov_e_circle_topology_mpi::topology MPITask(taskDataPar);
    ASSERT_EQ(MPITask.validation(), false);
  }
}

TEST(dormidontov_e_circle_topology_mpi, Test_if_ans_neq) {
  int size = 0;
  boost::mpi::communicator world;
  std::vector<int> mas;
  mas = dormidontov_e_circle_topology_mpi::GenVector(size);
  std::vector<int> ans(size + 1);
  std::vector<int> marks(world.size() + 1);
  std::vector<int> marks_of_ans = dormidontov_e_circle_topology_mpi::GenMarksAns(world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(marks.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.rank() + 1);
    dormidontov_e_circle_topology_mpi::topology MPITask(taskDataPar);
    ASSERT_EQ(MPITask.validation(), false);
  }
}

TEST(dormidontov_e_circle_topology_mpi, Test_2_in_power) {
  int size = 256;
  boost::mpi::communicator world;
  std::vector<int> mas;
  mas = dormidontov_e_circle_topology_mpi::GenVector(size);
  std::vector<int> ans(size);
  std::vector<int> marks(world.size() + 1);
  std::vector<int> marks_of_ans = dormidontov_e_circle_topology_mpi::GenMarksAns(world.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(marks.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.rank() + 1);
  }
  dormidontov_e_circle_topology_mpi::topology MPITask(taskDataPar);
  ASSERT_EQ(MPITask.validation(), true);
  MPITask.pre_processing();
  MPITask.run();
  MPITask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(mas, ans);
    ASSERT_EQ(marks, marks_of_ans);
  }
}

TEST(dormidontov_e_circle_topology_mpi, Test_3_in_power) {
  int size = 243;
  boost::mpi::communicator world;
  std::vector<int> mas;
  mas = dormidontov_e_circle_topology_mpi::GenVector(size);
  std::vector<int> ans(size);
  std::vector<int> marks(world.size() + 1);
  std::vector<int> marks_of_ans = dormidontov_e_circle_topology_mpi::GenMarksAns(world.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(marks.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.rank() + 1);
  }
  dormidontov_e_circle_topology_mpi::topology MPITask(taskDataPar);
  ASSERT_EQ(MPITask.validation(), true);
  MPITask.pre_processing();
  MPITask.run();
  MPITask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(mas, ans);
    ASSERT_EQ(marks, marks_of_ans);
  }
}

TEST(dormidontov_e_circle_topology_mpi, Test_primes) {
  int size = 563;
  boost::mpi::communicator world;
  std::vector<int> mas;
  mas = dormidontov_e_circle_topology_mpi::GenVector(size);
  std::vector<int> ans(size);
  std::vector<int> marks(world.size() + 1);
  std::vector<int> marks_of_ans = dormidontov_e_circle_topology_mpi::GenMarksAns(world.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(marks.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.rank() + 1);
  }
  dormidontov_e_circle_topology_mpi::topology MPITask(taskDataPar);
  ASSERT_EQ(MPITask.validation(), true);
  MPITask.pre_processing();
  MPITask.run();
  MPITask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(mas, ans);
    ASSERT_EQ(marks, marks_of_ans);
  }
}