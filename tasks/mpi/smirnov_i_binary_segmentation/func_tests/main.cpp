#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/smirnov_i_binary_segmentation/include/ops_mpi.hpp"

TEST(smirnov_i_binary_segmentation_mpi, negative_sizes_img) {
  boost::mpi::communicator world;
  int cols = -1;
  int rows = 1;
  std::vector<int> img;
  std::vector<int> mask;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    img = std::vector<int>(1, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
  }

  smirnov_i_binary_segmentation::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}
TEST(smirnov_i_binary_segmentation_mpi, not_binary_img) {
  boost::mpi::communicator world;
  int cols = 1;
  int rows = 1;
  std::vector<int> img;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    img = std::vector<int>(cols * rows, 3);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
  }

  smirnov_i_binary_segmentation::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}
TEST(smirnov_i_binary_segmentation_mpi, get_mask_for_scalar) {
  boost::mpi::communicator world;
  int cols = 1;
  int rows = 1;
  std::vector<int> img;
  std::vector<int> mask;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    img = std::vector<int>(cols * rows, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);

    mask = std::vector<int>(cols * rows, 1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
    taskDataPar->outputs_count.emplace_back(cols);
    taskDataPar->outputs_count.emplace_back(rows);
  }

  smirnov_i_binary_segmentation::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));

    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    std::vector<int> mask_seq(cols * rows, 1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask_seq.data()));
    taskDataSeq->outputs_count.emplace_back(cols);
    taskDataSeq->outputs_count.emplace_back(rows);

    auto TestTaskSequential = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskSequential>(taskDataSeq);

    ASSERT_EQ(TestTaskSequential->validation(), true);
    TestTaskSequential->pre_processing();
    TestTaskSequential->run();
    TestTaskSequential->post_processing();

    for (int i = 0; i < rows * cols; i++) {
      ASSERT_EQ(mask_seq[i], mask[i]);
    }
  }
}
TEST(smirnov_i_binary_segmentation_mpi, get_mask_for_vector) {
  boost::mpi::communicator world;
  int cols = 10;
  int rows = 1;
  std::vector<int> img;
  std::vector<int> mask;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    img = std::vector<int>(cols * rows, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);

    mask = std::vector<int>(cols * rows, 1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
    taskDataPar->outputs_count.emplace_back(cols);
    taskDataPar->outputs_count.emplace_back(rows);
  }

  smirnov_i_binary_segmentation::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));

    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    std::vector<int> mask_seq(cols * rows, 1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask_seq.data()));
    taskDataSeq->outputs_count.emplace_back(cols);
    taskDataSeq->outputs_count.emplace_back(rows);

    auto TestTaskSequential = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskSequential>(taskDataSeq);

    ASSERT_EQ(TestTaskSequential->validation(), true);
    TestTaskSequential->pre_processing();
    TestTaskSequential->run();
    TestTaskSequential->post_processing();

    for (int i = 0; i < rows * cols; i++) {
      ASSERT_EQ(mask_seq[i], mask[i]);
    }
  }
}
TEST(smirnov_i_binary_segmentation_mpi, get_mask_small) {
  boost::mpi::communicator world;
  int cols = 8;
  int rows = 4;
  std::vector<int> img;
  std::vector<int> mask;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    img = {0,   255, 0, 0,   255, 255, 0,   0,   255, 255, 255, 0, 255, 255, 0,   255,
           255, 0,   0, 255, 255, 255, 255, 255, 255, 0,   255, 0, 255, 0,   255, 255};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);

    mask = std::vector<int>(cols * rows, 1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
    taskDataPar->outputs_count.emplace_back(cols);
    taskDataPar->outputs_count.emplace_back(rows);
  }

  smirnov_i_binary_segmentation::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));

    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    std::vector<int> mask_seq(cols * rows, 1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask_seq.data()));
    taskDataSeq->outputs_count.emplace_back(cols);
    taskDataSeq->outputs_count.emplace_back(rows);

    auto TestTaskSequential = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskSequential>(taskDataSeq);

    ASSERT_EQ(TestTaskSequential->validation(), true);
    TestTaskSequential->pre_processing();
    TestTaskSequential->run();
    TestTaskSequential->post_processing();

    for (int i = 0; i < rows * cols; i++) {
      ASSERT_EQ(mask_seq[i], mask[i]);
    }
  }
}

TEST(smirnov_i_binary_segmentation_mpi, get_mask_sizes_10_pow_and_2_pow) {
  boost::mpi::communicator world;
  int cols = 100;
  int rows = 16;
  std::vector<int> img;
  std::vector<int> mask;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 1);

  if (world.rank() == 0) {
    img = std::vector<int>(cols * rows);

    for (int i = 0; i < cols * rows; i++) {
      img[i] = dis(gen) * 255;
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);

    mask = std::vector<int>(cols * rows, 1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
    taskDataPar->outputs_count.emplace_back(cols);
    taskDataPar->outputs_count.emplace_back(rows);
  }

  smirnov_i_binary_segmentation::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));

    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    std::vector<int> mask_seq(cols * rows, 1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask_seq.data()));
    taskDataSeq->outputs_count.emplace_back(cols);
    taskDataSeq->outputs_count.emplace_back(rows);

    auto TestTaskSequential = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskSequential>(taskDataSeq);

    ASSERT_EQ(TestTaskSequential->validation(), true);
    TestTaskSequential->pre_processing();
    TestTaskSequential->run();
    TestTaskSequential->post_processing();

    for (int i = 0; i < rows * cols; i++) {
      ASSERT_EQ(mask_seq[i], mask[i]);
    }
  }
}
TEST(smirnov_i_binary_segmentation_mpi, get_mask_prime_size) {
  boost::mpi::communicator world;
  int cols = 89;
  int rows = 7;
  std::vector<int> img;
  std::vector<int> mask;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 1);

  if (world.rank() == 0) {
    img = std::vector<int>(cols * rows);

    for (int i = 0; i < cols * rows; i++) {
      img[i] = dis(gen) * 255;
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);

    mask = std::vector<int>(cols * rows, 1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
    taskDataPar->outputs_count.emplace_back(cols);
    taskDataPar->outputs_count.emplace_back(rows);
  }

  smirnov_i_binary_segmentation::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));

    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    std::vector<int> mask_seq(cols * rows, 1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask_seq.data()));
    taskDataSeq->outputs_count.emplace_back(cols);
    taskDataSeq->outputs_count.emplace_back(rows);

    auto TestTaskSequential = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskSequential>(taskDataSeq);

    ASSERT_EQ(TestTaskSequential->validation(), true);
    TestTaskSequential->pre_processing();
    TestTaskSequential->run();
    TestTaskSequential->post_processing();

    for (int i = 0; i < rows * cols; i++) {
      ASSERT_EQ(mask_seq[i], mask[i]);
    }
  }
}