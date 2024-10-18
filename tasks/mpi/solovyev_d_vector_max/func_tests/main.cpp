#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/solovyev_d_vector_max/include/header.hpp"

TEST(solovyev_d_vector_max_mpi, Test_Max) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_max(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 240;
    global_vec = solovyev_d_vector_max_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  solovyev_d_vector_max_mpi::VectorMaxMPIParallel VectorMaxMPIParallel(taskDataPar);
  ASSERT_EQ(VectorMaxMPIParallel.validation(), true);
  VectorMaxMPIParallel.pre_processing();
  VectorMaxMPIParallel.run();
  VectorMaxMPIParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    solovyev_d_vector_max_mpi::VectorMaxSequential VectorMaxMPISequential(taskDataSeq);
    ASSERT_EQ(VectorMaxMPISequential.validation(), true);
    VectorMaxMPISequential.pre_processing();
    VectorMaxMPISequential.run();
    VectorMaxMPISequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(solovyev_d_vector_max_mpi, Test_Max_2) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_max(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;
    global_vec = solovyev_d_vector_max_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  solovyev_d_vector_max_mpi::VectorMaxMPIParallel VectorMaxMPIParallel(taskDataPar);
  ASSERT_EQ(VectorMaxMPIParallel.validation(), true);
  VectorMaxMPIParallel.pre_processing();
  VectorMaxMPIParallel.run();
  VectorMaxMPIParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    solovyev_d_vector_max_mpi::VectorMaxSequential VectorMaxMPISequential(taskDataSeq);
    ASSERT_EQ(VectorMaxMPISequential.validation(), true);
    VectorMaxMPISequential.pre_processing();
    VectorMaxMPISequential.run();
    VectorMaxMPISequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}
