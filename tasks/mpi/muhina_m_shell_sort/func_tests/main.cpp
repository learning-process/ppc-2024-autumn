// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/muhina_m_shell_sort/include/ops_mpi.hpp"
namespace muhina_m_shell_sort_mpi {

std::vector<int> Get_Random_Vector(int sz, int min_value, int max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min_value + gen() % (max_value - min_value + 1);
  }
  return vec;
}
}  // namespace muhina_m_shell_sort_mpi
TEST(muhina_m_shell_sort, Test_Sort_SmallVec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {5, 2, 9, 1, 5, 6};
    out.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  muhina_m_shell_sort_mpi::ShellSortMPIParallel ShellSortMPIParalle(taskDataPar);
  ASSERT_EQ(ShellSortMPIParalle.validation(), true);
  ShellSortMPIParalle.pre_processing();
  ShellSortMPIParalle.run();
  ShellSortMPIParalle.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_min(global_vec.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    muhina_m_shell_sort_mpi::ShellSortMPISequential ShellSortMPISequential(taskDataSeq);
    ASSERT_EQ(ShellSortMPISequential.validation(), true);
    ShellSortMPISequential.pre_processing();
    ShellSortMPISequential.run();
    ShellSortMPISequential.post_processing();

    ASSERT_EQ(reference_min, out);
  }
}

TEST(muhina_m_shell_sort, Test_Sort_SingleElementVec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {5};
    out.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  muhina_m_shell_sort_mpi::ShellSortMPIParallel ShellSortMPIParalle(taskDataPar);
  ASSERT_EQ(ShellSortMPIParalle.validation(), true);
  ShellSortMPIParalle.pre_processing();
  ShellSortMPIParalle.run();
  ShellSortMPIParalle.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_min(global_vec.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    muhina_m_shell_sort_mpi::ShellSortMPISequential ShellSortMPISequential(taskDataSeq);
    ASSERT_EQ(ShellSortMPISequential.validation(), true);
    ShellSortMPISequential.pre_processing();
    ShellSortMPISequential.run();
    ShellSortMPISequential.post_processing();

    ASSERT_EQ(reference_min, out);
  }
}

TEST(muhina_m_shell_sort, Test_Sort_EmptyVec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  muhina_m_shell_sort_mpi::ShellSortMPIParallel ShellSortMPIParalle(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(ShellSortMPIParalle.validation());
  }
}

TEST(muhina_m_shell_sort, Test_Sort_LargeVector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 100;
    const int min_val = 0;
    const int max_val = 100;
    global_vec = muhina_m_shell_sort_mpi::Get_Random_Vector(count, min_val, max_val);
    out.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  muhina_m_shell_sort_mpi::ShellSortMPIParallel ShellSortMPIParalle(taskDataPar);
  ASSERT_EQ(ShellSortMPIParalle.validation(), true);
  ShellSortMPIParalle.pre_processing();
  ShellSortMPIParalle.run();
  ShellSortMPIParalle.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_min(global_vec.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    muhina_m_shell_sort_mpi::ShellSortMPISequential ShellSortMPISequential(taskDataSeq);
    ASSERT_EQ(ShellSortMPISequential.validation(), true);
    ShellSortMPISequential.pre_processing();
    ShellSortMPISequential.run();
    ShellSortMPISequential.post_processing();

    ASSERT_EQ(reference_min, out);
  }
}

TEST(muhina_m_shell_sort, Test_Sort_NegativeValues) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {-5, -2, -9, -1, -5, -6};
    out.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  muhina_m_shell_sort_mpi::ShellSortMPIParallel ShellSortMPIParalle(taskDataPar);
  ASSERT_EQ(ShellSortMPIParalle.validation(), true);
  ShellSortMPIParalle.pre_processing();
  ShellSortMPIParalle.run();
  ShellSortMPIParalle.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_min(global_vec.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    muhina_m_shell_sort_mpi::ShellSortMPISequential ShellSortMPISequential(taskDataSeq);
    ASSERT_EQ(ShellSortMPISequential.validation(), true);
    ShellSortMPISequential.pre_processing();
    ShellSortMPISequential.run();
    ShellSortMPISequential.post_processing();

    ASSERT_EQ(reference_min, out);
  }
}

TEST(muhina_m_shell_sort, Test_Sort_RepeatingValues) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {2, 2, 2, 2, 2, 2};
    out.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  muhina_m_shell_sort_mpi::ShellSortMPIParallel ShellSortMPIParalle(taskDataPar);
  ASSERT_EQ(ShellSortMPIParalle.validation(), true);
  ShellSortMPIParalle.pre_processing();
  ShellSortMPIParalle.run();
  ShellSortMPIParalle.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_min(global_vec.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    muhina_m_shell_sort_mpi::ShellSortMPISequential ShellSortMPISequential(taskDataSeq);
    ASSERT_EQ(ShellSortMPISequential.validation(), true);
    ShellSortMPISequential.pre_processing();
    ShellSortMPISequential.run();
    ShellSortMPISequential.post_processing();

    ASSERT_EQ(reference_min, out);
  }
}

TEST(muhina_m_shell_sort, Test_Sort_10_Elements) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 10;
    const int min_val = -100;
    const int max_val = 100;
    global_vec = muhina_m_shell_sort_mpi::Get_Random_Vector(count, min_val, max_val);
    out.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  muhina_m_shell_sort_mpi::ShellSortMPIParallel ShellSortMPIParalle(taskDataPar);
  ASSERT_EQ(ShellSortMPIParalle.validation(), true);
  ShellSortMPIParalle.pre_processing();
  ShellSortMPIParalle.run();
  ShellSortMPIParalle.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_min(global_vec.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    muhina_m_shell_sort_mpi::ShellSortMPISequential ShellSortMPISequential(taskDataSeq);
    ASSERT_EQ(ShellSortMPISequential.validation(), true);
    ShellSortMPISequential.pre_processing();
    ShellSortMPISequential.run();
    ShellSortMPISequential.post_processing();

    ASSERT_EQ(reference_min, out);
  }
}

TEST(muhina_m_shell_sort, Test_Sort_20_Elements) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 20;
    const int min_val = -100;
    const int max_val = 100;
    global_vec = muhina_m_shell_sort_mpi::Get_Random_Vector(count, min_val, max_val);
    out.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  muhina_m_shell_sort_mpi::ShellSortMPIParallel ShellSortMPIParalle(taskDataPar);
  ASSERT_EQ(ShellSortMPIParalle.validation(), true);
  ShellSortMPIParalle.pre_processing();
  ShellSortMPIParalle.run();
  ShellSortMPIParalle.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_min(global_vec.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    muhina_m_shell_sort_mpi::ShellSortMPISequential ShellSortMPISequential(taskDataSeq);
    ASSERT_EQ(ShellSortMPISequential.validation(), true);
    ShellSortMPISequential.pre_processing();
    ShellSortMPISequential.run();
    ShellSortMPISequential.post_processing();

    ASSERT_EQ(reference_min, out);
  }
}
TEST(muhina_m_shell_sort, Test_Sort_50_Elements) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 50;
    const int min_val = -100;
    const int max_val = 100;
    global_vec = muhina_m_shell_sort_mpi::Get_Random_Vector(count, min_val, max_val);
    out.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  muhina_m_shell_sort_mpi::ShellSortMPIParallel ShellSortMPIParalle(taskDataPar);
  ASSERT_EQ(ShellSortMPIParalle.validation(), true);
  ShellSortMPIParalle.pre_processing();
  ShellSortMPIParalle.run();
  ShellSortMPIParalle.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_min(global_vec.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    muhina_m_shell_sort_mpi::ShellSortMPISequential ShellSortMPISequential(taskDataSeq);
    ASSERT_EQ(ShellSortMPISequential.validation(), true);
    ShellSortMPISequential.pre_processing();
    ShellSortMPISequential.run();
    ShellSortMPISequential.post_processing();

    ASSERT_EQ(reference_min, out);
  }
}
TEST(muhina_m_shell_sort, Test_Sort_150_Elements) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 150;
    const int min_val = -100;
    const int max_val = 100;
    global_vec = muhina_m_shell_sort_mpi::Get_Random_Vector(count, min_val, max_val);
    out.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  muhina_m_shell_sort_mpi::ShellSortMPIParallel ShellSortMPIParalle(taskDataPar);
  ASSERT_EQ(ShellSortMPIParalle.validation(), true);
  ShellSortMPIParalle.pre_processing();
  ShellSortMPIParalle.run();
  ShellSortMPIParalle.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_min(global_vec.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    muhina_m_shell_sort_mpi::ShellSortMPISequential ShellSortMPISequential(taskDataSeq);
    ASSERT_EQ(ShellSortMPISequential.validation(), true);
    ShellSortMPISequential.pre_processing();
    ShellSortMPISequential.run();
    ShellSortMPISequential.post_processing();

    ASSERT_EQ(reference_min, out);
  }
}