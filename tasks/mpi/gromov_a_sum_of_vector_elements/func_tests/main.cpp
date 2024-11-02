#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/gromov_a_sum_of_vector_elements/include/ops_mpi.hpp"

TEST(gromov_a_sum_of_vector_elements_mpi, Test_Production) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_production(1, 1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 50;
    global_vec = gromov_a_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_production.data()));
    taskDataPar->outputs_count.emplace_back(global_production.size());
  }

  gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel MPISumOfVectorParallel(taskDataPar, "production");
  ASSERT_EQ(MPISumOfVectorParallel.validation(), true);
  MPISumOfVectorParallel.pre_processing();
  MPISumOfVectorParallel.run();
  MPISumOfVectorParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_production(1, 1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_production.data()));
    taskDataSeq->outputs_count.emplace_back(reference_production.size());

    // Create Task
    gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential MPISumOfVectorSequential(taskDataSeq, "production");
    ASSERT_EQ(MPISumOfVectorSequential.validation(), true);
    MPISumOfVectorSequential.pre_processing();
    MPISumOfVectorSequential.run();
    MPISumOfVectorSequential.post_processing();

    ASSERT_EQ(reference_production[0], global_production[0]);
  }
}

TEST(gromov_a_sum_of_vector_elements_mpi, Test_Min) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, std::numeric_limits<int32_t>::max());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = gromov_a_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel MPISumOfVectorParallel(taskDataPar, "min");
  ASSERT_EQ(MPISumOfVectorParallel.validation(), true);
  MPISumOfVectorParallel.pre_processing();
  MPISumOfVectorParallel.run();
  MPISumOfVectorParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, std::numeric_limits<int32_t>::max());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    // Create Task
    gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential MPISumOfVectorSequential(taskDataSeq, "min");
    ASSERT_EQ(MPISumOfVectorSequential.validation(), true);
    MPISumOfVectorSequential.pre_processing();
    MPISumOfVectorSequential.run();
    MPISumOfVectorSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(gromov_a_sum_of_vector_elements_mpi, Test_Max) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_max(1, std::numeric_limits<int32_t>::min());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = gromov_a_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel MPISumOfVectorParallel(taskDataPar, "max");
  ASSERT_EQ(MPISumOfVectorParallel.validation(), true);
  MPISumOfVectorParallel.pre_processing();
  MPISumOfVectorParallel.run();
  MPISumOfVectorParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, std::numeric_limits<int32_t>::min());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential MPISumOfVectorSequential(taskDataSeq, "max");
    ASSERT_EQ(MPISumOfVectorSequential.validation(), true);
    MPISumOfVectorSequential.pre_processing();
    MPISumOfVectorSequential.run();
    MPISumOfVectorSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(gromov_a_sum_of_vector_elements_mpi, Test_Max2) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_max(1, std::numeric_limits<int32_t>::min());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 200;
    global_vec = gromov_a_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel MPISumOfVectorParallel(taskDataPar, "max");
  ASSERT_EQ(MPISumOfVectorParallel.validation(), true);
  MPISumOfVectorParallel.pre_processing();
  MPISumOfVectorParallel.run();
  MPISumOfVectorParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, std::numeric_limits<int32_t>::min());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential MPISumOfVectorSequential(taskDataSeq, "max");
    ASSERT_EQ(MPISumOfVectorSequential.validation(), true);
    MPISumOfVectorSequential.pre_processing();
    MPISumOfVectorSequential.run();
    MPISumOfVectorSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

    TEST(gromov_a_sum_of_vector_elements_mpi, Test_Addition) {
        boost::mpi::communicator world;
        std::vector<int> global_vec;
        std::vector<int32_t> global_add(1, 0);
        // Create TaskData
        std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

        if (world.rank() == 0) {
            const int count_size_vector = 200;
            global_vec = gromov_a_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
            taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
            taskDataPar->inputs_count.emplace_back(global_vec.size());
            taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_add.data()));
            taskDataPar->outputs_count.emplace_back(global_add.size());
        }

        gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel MPISumOfVectorParallel(taskDataPar, "add");
        ASSERT_EQ(MPISumOfVectorParallel.validation(), true);
        MPISumOfVectorParallel.pre_processing();
        MPISumOfVectorParallel.run();
        MPISumOfVectorParallel.post_processing();

        if (world.rank() == 0) {
            // Create data
            std::vector<int32_t> reference_add(1, 0);

            // Create TaskData
            std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
            taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
            taskDataSeq->inputs_count.emplace_back(global_vec.size());
            taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_add.data()));
            taskDataSeq->outputs_count.emplace_back(reference_add.size());

            // Create Task
            gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential MPISumOfVectorSequential(taskDataSeq, "add");
            ASSERT_EQ(MPISumOfVectorSequential.validation(), true);
            MPISumOfVectorSequential.pre_processing();
            MPISumOfVectorSequential.run();
            MPISumOfVectorSequential.post_processing();

            ASSERT_EQ(reference_add[0], global_add[0]);
        }
    }

    TEST(gromov_a_sum_of_vector_elements_mpi, Test_Subtraction) {
        boost::mpi::communicator world;
        std::vector<int> global_vec;
        std::vector<int32_t> global_sub(1, 0);
        // Create TaskData
        std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

        if (world.rank() == 0) {
            const int count_size_vector = 240;
            global_vec = gromov_a_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
            taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
            taskDataPar->inputs_count.emplace_back(global_vec.size());
            taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sub.data()));
            taskDataPar->outputs_count.emplace_back(global_sub.size());
        }

        gromov_a_sum_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "sub");
        ASSERT_EQ(testMpiTaskParallel.validation(), true);
        testMpiTaskParallel.pre_processing();
        testMpiTaskParallel.run();
        testMpiTaskParallel.post_processing();

        if (world.rank() == 0) {
            // Create data
            std::vector<int32_t> reference_sub(1, 0);

            // Create TaskData
            std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
            taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
            taskDataSeq->inputs_count.emplace_back(global_vec.size());
            taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sub.data()));
            taskDataSeq->outputs_count.emplace_back(reference_sub.size());

            // Create Task
            gromov_a_sum_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "sub");
            ASSERT_EQ(testMpiTaskSequential.validation(), true);
            testMpiTaskSequential.pre_processing();
            testMpiTaskSequential.run();
            testMpiTaskSequential.post_processing();

            ASSERT_EQ(reference_sub[0], global_sub[0]);
        }
    }