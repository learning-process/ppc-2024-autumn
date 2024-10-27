
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kholin_k_vector_neighbor_diff_elems/include/ops_mpi.hpp"

TEST(kholin_k_vector_neighbor_diff_elems_mpi, check_validation) {
  boost::mpi::communicator world;
  const int count_size_vector = 500;
  std::vector<int> global_vec;
  std::vector<double> global_delta(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = std::vector<int>(count_size_vector);

    global_vec[100] = 5000;
    global_vec[101] = 1;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_delta.data()));
    taskDataPar->outputs_count.emplace_back(global_delta.size());
  }

  kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<int> testMpiTaskParallel(taskDataPar,
                                                                                                  "MAX_DIFFERENCE");
  bool IsValid = testMpiTaskParallel.validation();
  ASSERT_EQ(IsValid, true);

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_delta(1, 0);
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
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_delta.data()));
    taskDataSeq->outputs_count.emplace_back(reference_delta.size());

    // Create Task
    kholin_k_vector_neighbor_diff_elems_mpi::TestTaskSequential<int, uint64_t> testMPITaskSequential(taskDataSeq,
                                                                                                     "MAX_DIFFERENCE");
    bool IsValid_ = testMPITaskSequential.validation();
    ASSERT_EQ(IsValid_, true);
  }
}

TEST(kholin_k_vector_neighbor_diff_elems_mpi, check_pre_processing) {
  boost::mpi::communicator world;
  const int count_size_vector = 500;
  std::vector<int> global_vec;
  std::vector<double> global_delta(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = std::vector<int>(count_size_vector);

    global_vec[100] = 5000;
    global_vec[101] = 1;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_delta.data()));
    taskDataPar->outputs_count.emplace_back(global_delta.size());
  }

  kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<int> testMpiTaskParallel(taskDataPar,
                                                                                                  "MAX_DIFFERENCE");
  testMpiTaskParallel.validation();
  bool IsValid = testMpiTaskParallel.pre_processing();
  ASSERT_EQ(IsValid, true);

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_delta(1, 0);
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
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_delta.data()));
    taskDataSeq->outputs_count.emplace_back(reference_delta.size());

    // Create Task
    kholin_k_vector_neighbor_diff_elems_mpi::TestTaskSequential<int, uint64_t> testTaskSequential(taskDataSeq,
                                                                                                  "MAX_DIFFERENCE");
    testTaskSequential.validation();
    bool IsValid_ = testTaskSequential.pre_processing();
    ASSERT_EQ(IsValid_, true);
  }
}

TEST(kholin_k_vector_neighbor_diff_elems_mpi, check_run) {
  boost::mpi::communicator world;
  const int count_size_vector = 150;
  std::vector<int> global_vec;
  std::vector<double> global_delta(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = std::vector<int>(count_size_vector);
    for (size_t i = 0; i < global_vec.size(); i++) {
      global_vec[i] = 4 * i + 2;
    }
    global_vec[100] = 5000;
    global_vec[101] = 1;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_delta.data()));
    taskDataPar->outputs_count.emplace_back(global_delta.size());
  }

  kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<int> testMpiTaskParallel(taskDataPar,
                                                                                                  "MAX_DIFFERENCE");
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  bool IsValid = testMpiTaskParallel.run();
  ASSERT_EQ(IsValid, true);

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_delta(1, 0);
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
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_delta.data()));
    taskDataSeq->outputs_count.emplace_back(reference_delta.size());

    // Create Task
    kholin_k_vector_neighbor_diff_elems_mpi::TestTaskSequential<int> testTaskSequential(taskDataSeq,
                                                                                                  "MAX_DIFFERENCE");
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    bool IsValid_ = testTaskSequential.run();
    ASSERT_EQ(IsValid_, true);
  }
}

TEST(kholin_k_vector_neighbor_diff_elems_mpi, check_post_processing) {
  boost::mpi::communicator world;
  const int count_size_vector = 500;
  std::vector<int> global_vec;
  std::vector<double> global_delta(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = std::vector<int>(count_size_vector);
    for (size_t i = 0; i < global_vec.size(); i++) {
      global_vec[i] = 4 * i + 2;
    }
    global_vec[100] = 5000;
    global_vec[101] = 1;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_delta.data()));
    taskDataPar->outputs_count.emplace_back(global_delta.size());
  }

  kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<int> testMpiTaskParallel(taskDataPar,
                                                                                                  "MAX_DIFFERENCE");
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  bool IsValid = testMpiTaskParallel.post_processing();
  ASSERT_EQ(IsValid, true);

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_delta(1, 0);
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
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_delta.data()));
    taskDataSeq->outputs_count.emplace_back(reference_delta.size());

    // Create Task
    kholin_k_vector_neighbor_diff_elems_mpi::TestTaskSequential<int, uint64_t> testTaskSequential(taskDataSeq,
                                                                                                  "MAX_DIFFERENCE");
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    bool IsValid_ = testTaskSequential.post_processing();
    ASSERT_EQ(IsValid_, true);
  }
}

TEST(kholin_k_vector_neighbor_diff_elems_mpi, check_int) {
  boost::mpi::communicator world;
  const int count_size_vector = 200;
  std::vector<int> global_vec;
  std::vector<double> global_delta(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = std::vector<int>(count_size_vector);
    for (size_t i = 0; i < global_vec.size(); i++) {
      global_vec[i] = 4 * i + 2;
    }

    global_vec[100] = 5000;
    global_vec[101] = 1;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_delta.data()));
    taskDataPar->outputs_count.emplace_back(global_delta.size());
  }

  kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<int> testMpiTaskParallel(taskDataPar,
                                                                                                  "MAX_DIFFERENCE");
  bool IsValid = testMpiTaskParallel.validation();
  ASSERT_EQ(IsValid, true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  int test = global_delta[0];

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_delta(1, 0);
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
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_delta.data()));
    taskDataSeq->outputs_count.emplace_back(reference_delta.size());

    // Create Task
    kholin_k_vector_neighbor_diff_elems_mpi::TestTaskSequential<int, uint64_t> testTaskSequential(taskDataSeq,
                                                                                                  "MAX_DIFFERENCE");
    bool IsValid_ = testTaskSequential.validation();
    ASSERT_EQ(IsValid_, true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    int test2 = reference_elems[0] - reference_elems[1];
    ASSERT_EQ(test, test2);
  }
}
TEST(kholin_k_vector_neighbor_diff_elems_mpi, check_int32_t) {
  boost::mpi::communicator world;
  std::vector<int32_t> global_vec;
  std::vector<double> global_delta(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 300;
    global_vec = std::vector<int32_t>(count_size_vector);
    for (size_t i = 0; i < global_vec.size(); i++) {
      global_vec[i] = 2 * i + 4;
    }
    global_vec[100] = 5000;
    global_vec[101] = 1;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_delta.data()));
    taskDataPar->outputs_count.emplace_back(global_delta.size());
  }

  kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<int32_t> testMpiTaskParallel(taskDataPar,
                                                                                                      "MAX_DIFFERENCE");
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  int32_t test = global_delta[0];

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_delta(1, 0);
    std::vector<int32_t> reference_elems(2, 0);
    std::vector<uint64_t> reference_indices(2, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_elems.data()));
    taskDataSeq->outputs_count.emplace_back(reference_elems.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_indices.data()));
    taskDataSeq->outputs_count.emplace_back(reference_indices.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_delta.data()));
    taskDataSeq->outputs_count.emplace_back(reference_delta.size());

    // Create Task
    kholin_k_vector_neighbor_diff_elems_mpi::TestTaskSequential<int32_t, uint64_t> testTaskSequential(taskDataSeq,
                                                                                                      "MAX_DIFFERENCE");
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    int32_t test2 = reference_elems[0] - reference_elems[1];
    ASSERT_EQ(test, test2);
  }
}

TEST(kholin_k_vector_neighbor_diff_elems_mpi, check_float) {
  boost::mpi::communicator world;
  std::vector<float> global_vec;
  std::vector<double> global_delta(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 1000;
    global_vec = std::vector<float>(count_size_vector);
    for (size_t i = 0; i < global_vec.size(); i++) {
      global_vec[i] = 0.25 * i + 10;
    }

    global_vec[100] = 110.001f;
    global_vec[101] = -990.0025f;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_delta.data()));
    taskDataPar->outputs_count.emplace_back(global_delta.size());
  }

  kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<float> testMpiTaskParallel(taskDataPar,
                                                                                                    "MAX_DIFFERENCE");
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  float test = global_delta[0];

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_delta(1, 0);
    std::vector<float> reference_elems(2, 0);
    std::vector<uint64_t> reference_indices(2, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_elems.data()));
    taskDataSeq->outputs_count.emplace_back(reference_elems.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_indices.data()));
    taskDataSeq->outputs_count.emplace_back(reference_indices.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_delta.data()));
    taskDataSeq->outputs_count.emplace_back(reference_delta.size());

    // Create Task
    kholin_k_vector_neighbor_diff_elems_mpi::TestTaskSequential<float, uint64_t> testTaskSequential(taskDataSeq,
                                                                                                    "MAX_DIFFERENCE");
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    float test2 = reference_elems[0] - reference_elems[1];
    ASSERT_NEAR(test, test2, 1e-5);
  }
}

TEST(kholin_k_vector_neighbor_diff_elems_mpi, check_double) {
  boost::mpi::communicator world;
  std::vector<double> global_vec;
  std::vector<double> global_delta(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 750;
    global_vec = std::vector<double>(count_size_vector);
    for (size_t i = 0; i < global_vec.size(); i++) {
      global_vec[i] = 0.25 * i + 10;
    }

    global_vec[100] = 110.001;
    global_vec[101] = -990.0025;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_delta.data()));
    taskDataPar->outputs_count.emplace_back(global_delta.size());
  }

  kholin_k_vector_neighbor_diff_elems_mpi::TestMPITaskParallel<double> testMpiTaskParallel(taskDataPar,
                                                                                                     "MAX_DIFFERENCE");
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  double test = global_delta[0];

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_delta(1, 0);
    std::vector<double> reference_elems(2, 0);
    std::vector<uint64_t> reference_indices(2, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_elems.data()));
    taskDataSeq->outputs_count.emplace_back(reference_elems.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_indices.data()));
    taskDataSeq->outputs_count.emplace_back(reference_indices.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_delta.data()));
    taskDataSeq->outputs_count.emplace_back(reference_delta.size());

    // Create Task
    kholin_k_vector_neighbor_diff_elems_mpi::TestTaskSequential<double, uint64_t> testTaskSequential(taskDataSeq,
                                                                                                     "MAX_DIFFERENCE");
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    double test2 = reference_elems[0] - reference_elems[1];
    ASSERT_NEAR(test, test2, 1e-5);
  }
}