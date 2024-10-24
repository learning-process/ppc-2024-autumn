
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kholin_k_vector_neighbor_diff_elems/include/ops_mpi.hpp"

TEST(kholin_k_vector_neighbor_diff_elems_mpi, check_validation) {
  boost::mpi::communicator world;
  const int count_size_vector = 500;
  std::vector<int> global_vec;
  std::vector<int> global_elems(2, 0);
  std::vector<uint64_t> global_indices(2, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = std::vector<int>(count_size_vector);

    global_vec[100] = 5000;
    global_vec[101] = 1;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_elems.data()));
    taskDataPar->outputs_count.emplace_back(global_elems.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices.data()));
    taskDataPar->outputs_count.emplace_back(global_indices.size());
  }

  kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<int, uint64_t> testMpiTaskParallel(taskDataPar,
                                                                                                  "MAX_DIFFERENCE");
  bool IsValid = testMpiTaskParallel.validation();
  ASSERT_EQ(IsValid, true);

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_elems(2, 0);
    std::vector<uint64_t> reference_indices(2, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_elems.data()));
    taskDataSeq->outputs_count.emplace_back(reference_elems.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_indices.data()));
    taskDataSeq->outputs_count.emplace_back(reference_indices.size());

    // Create Task
    kholin_k_vector_neighbor_diff_elems_mpi::TestTaskSequential<int, uint64_t> testMPITaskSequential(taskDataSeq,
                                                                                                  "MAX_DIFFERENCE");
    bool IsValid_ = testMPITaskSequential.validation();
    ASSERT_EQ(IsValid_, true);
  }
}

//TEST(kholin_k_vector_neighbor_diff_elems_mpi, check_int) {
//  boost::mpi::communicator world;
//  std::vector<int> global_vec;
//  std::vector<int> global_elems(2, 0);
//  std::vector<uint64_t> global_indices(2, 0);
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    const int count_size_vector = 500;
//    global_vec = std::vector<int>(count_size_vector);
//    for (size_t i = 0; i < global_vec.size(); i++) {
//      global_vec[i] = 4 * i + 2;
//    }
//
//    global_vec[100] = 5000;
//    global_vec[101] = 1;
//
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataPar->inputs_count.emplace_back(global_vec.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_elems.data()));
//    taskDataPar->outputs_count.emplace_back(global_elems.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices.data()));
//    taskDataPar->outputs_count.emplace_back(global_indices.size());
//  }
//
//  kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<int, uint64_t> testMpiTaskParallel(taskDataPar,
//                                                                                                  "MAX_DIFFERENCE");
//  testMpiTaskParallel.validation();
//  testMpiTaskParallel.pre_processing();
//  testMpiTaskParallel.run();
//  testMpiTaskParallel.post_processing();
//  EXPECT_EQ(global_elems[0], 5000);
//  EXPECT_EQ(global_elems[1], 1);
//  EXPECT_EQ(global_indices[0], 100ull);
//  EXPECT_EQ(global_indices[1], 101ull);
//
//  if (world.rank() == 0) {
//    // Create data
//    std::vector<int> reference_elems(2, 0);
//    std::vector<uint64_t> reference_indices(2, 0);
//    // Create TaskData
//    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataSeq->inputs_count.emplace_back(global_vec.size());
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_elems.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_elems.size());
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_indices.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_indices.size());
//
//    // Create Task
//    kholin_k_vector_neighbor_diff_elems_mpi::TestTaskSequential<int, uint64_t> testTaskSequential(taskDataSeq,
//                                                                                                  "MAX_DIFFERENCE");
//    testTaskSequential.validation();
//    testTaskSequential.pre_processing();
//    testTaskSequential.run();
//    testTaskSequential.post_processing();
//    EXPECT_EQ(reference_elems[0], 5000);
//    EXPECT_EQ(reference_elems[1], 1);
//    EXPECT_EQ(reference_indices[0], 100ull);
//    EXPECT_EQ(reference_indices[1], 101ull);
//  }
//}
//TEST(kholin_k_vector_neighbor_diff_elems_mpi, check_int32_t) {
//  boost::mpi::communicator world;
//  std::vector<int32_t>global_vec;
//  std::vector<int32_t> global_elems(2, 0);
//  std::vector<uint64_t> global_indices(2, 0);
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//  if (world.rank() == 0) {
//    const int count_size_vector = 500;
//    global_vec = std::vector<int32_t>(count_size_vector);
//    for (size_t i = 0; i < global_vec.size(); i++) {
//      global_vec[i] = 2 * i + 4;
//    }
//    global_vec[100] = 5000;
//    global_vec[101] = 1;
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataPar->inputs_count.emplace_back(global_vec.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_elems.data()));
//    taskDataPar->outputs_count.emplace_back(global_elems.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices.data()));
//    taskDataPar->outputs_count.emplace_back(global_indices.size());
//  }
//
//  kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<int32_t, uint64_t> testMpiTaskParallel(taskDataPar,
//                                                                                                      "MAX_DIFFERENCE");
//  testMpiTaskParallel.validation();
//  testMpiTaskParallel.pre_processing();
//  testMpiTaskParallel.run();
//  testMpiTaskParallel.post_processing();
//  EXPECT_EQ(global_elems[0], 5000l);
//  EXPECT_EQ(global_elems[1], 1l);
//  EXPECT_EQ(global_indices[0], 100ull);
//  EXPECT_EQ(global_indices[1], 101ull);
//
//  if (world.rank() == 0) {
//    // Create data
//    std::vector<int32_t> reference_elems(2, 0);
//    std::vector<uint64_t> reference_indices(2, 0);
//
//    // Create TaskData
//    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataSeq->inputs_count.emplace_back(global_vec.size());
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_elems.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_elems.size());
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_indices.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_indices.size());
//
//    // Create Task
//    kholin_k_vector_neighbor_diff_elems_mpi::TestTaskSequential<int32_t, uint64_t> testTaskSequential(taskDataSeq,
//                                                                                                      "MAX_DIFFERENCE");
//    testTaskSequential.validation();
//    testTaskSequential.pre_processing();
//    testTaskSequential.run();
//    testTaskSequential.post_processing();
//    EXPECT_EQ(reference_elems[0], 5000l);
//    EXPECT_EQ(reference_elems[1], 1l);
//    EXPECT_EQ(reference_indices[0], 100ull);
//    EXPECT_EQ(reference_indices[1], 101ull);
//  }
//}
//
//TEST(kholin_k_vector_neighbor_diff_elems_mpi, check_float) {
//  boost::mpi::communicator world;
//  std::vector<float> global_vec;
//  std::vector<float> global_elems(2, 0);
//  std::vector<uint64_t> global_indices(2, 0);
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    const int count_size_vector = 500;
//    global_vec = std::vector<float>(count_size_vector);
//    for (size_t i = 0; i < global_vec.size(); i++) {
//      global_vec[i] = 0.25 * i + 10;
//    }
//
//    global_vec[100] = 110.001f;
//    global_vec[101] = -990.0025f;
//
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataPar->inputs_count.emplace_back(global_vec.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_elems.data()));
//    taskDataPar->outputs_count.emplace_back(global_elems.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices.data()));
//    taskDataPar->outputs_count.emplace_back(global_indices.size());
//  }
//
//  kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<float, uint64_t> testMpiTaskParallel(taskDataPar,
//                                                                                                    "MAX_DIFFERENCE");
//  testMpiTaskParallel.validation();
//  testMpiTaskParallel.pre_processing();
//  testMpiTaskParallel.run();
//  testMpiTaskParallel.post_processing();
//  EXPECT_NEAR(global_elems[0], 110.001f, 1e-4);
//  EXPECT_NEAR(global_elems[1], -990.0025f, 1e-4);
//  EXPECT_EQ(global_indices[0], 100ull);
//  EXPECT_EQ(global_indices[1], 101ull);
//
//  if (world.rank() == 0) {
//    // Create data
//    std::vector<float> reference_elems(2, 0);
//    std::vector<uint64_t> reference_indices(2, 0);
//
//    // Create TaskData
//    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataSeq->inputs_count.emplace_back(global_vec.size());
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_elems.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_elems.size());
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_indices.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_indices.size());
//
//    // Create Task
//    kholin_k_vector_neighbor_diff_elems_mpi::TestTaskSequential<float, uint64_t> testTaskSequential(taskDataSeq,
//                                                                                                    "MAX_DIFFERENCE");
//    testTaskSequential.validation();
//    testTaskSequential.pre_processing();
//    testTaskSequential.run();
//    testTaskSequential.post_processing();
//    EXPECT_NEAR(reference_elems[0], 110.001f, 1e-4);
//    EXPECT_NEAR(reference_elems[1], -990.0025f, 1e-4);
//    EXPECT_EQ(reference_indices[0], 100ull);
//    EXPECT_EQ(reference_indices[1], 101ull);
//  }
//}
//
//TEST(kholin_k_vector_neighbor_diff_elems_mpi, check_double) {
//  boost::mpi::communicator world;
//  std::vector<double> global_vec;
//  std::vector<double> global_elems(2, 0);
//  std::vector<uint64_t> global_indices(2, 0);
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    const int count_size_vector = 500;
//    global_vec = std::vector<double>(count_size_vector);
//    for (size_t i = 0; i < global_vec.size(); i++) {
//      global_vec[i] = 0.25 * i + 10;
//    }
//
//    global_vec[100] = 110.001;
//    global_vec[101] = -990.0025;
//
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataPar->inputs_count.emplace_back(global_vec.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_elems.data()));
//    taskDataPar->outputs_count.emplace_back(global_elems.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_indices.data()));
//    taskDataPar->outputs_count.emplace_back(global_indices.size());
//  }
//
//  kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<double, uint64_t> testMpiTaskParallel(taskDataPar,
//                                                                                                     "MAX_DIFFERENCE");
//  testMpiTaskParallel.validation();
//  testMpiTaskParallel.pre_processing();
//  testMpiTaskParallel.run();
//  testMpiTaskParallel.post_processing();
//  EXPECT_NEAR(global_elems[0], 110.001f, 1e-4);
//  EXPECT_NEAR(global_elems[1], -990.0025f, 1e-4);
//  EXPECT_EQ(global_indices[0], 100ull);
//  EXPECT_EQ(global_indices[1], 101ull);
//
//  if (world.rank() == 0) {
//    // Create data
//    std::vector<double> reference_elems(2, 0);
//    std::vector<uint64_t> reference_indices(2, 0);
//
//    // Create TaskData
//    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//    taskDataSeq->inputs_count.emplace_back(global_vec.size());
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_elems.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_elems.size());
//    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_indices.data()));
//    taskDataSeq->outputs_count.emplace_back(reference_indices.size());
//
//    // Create Task
//    kholin_k_vector_neighbor_diff_elems_mpi::TestTaskSequential<double, uint64_t> testTaskSequential(taskDataSeq,
//                                                                                                     "MAX_DIFFERENCE");
//    testTaskSequential.validation();
//    testTaskSequential.pre_processing();
//    testTaskSequential.run();
//    testTaskSequential.post_processing();
//    EXPECT_NEAR(reference_elems[0], 110.001, 1e-4);
//    EXPECT_NEAR(reference_elems[1], -990.0025, 1e-4);
//    EXPECT_EQ(reference_indices[0], 100ull);
//    EXPECT_EQ(reference_indices[1], 101ull);
//  }
//}