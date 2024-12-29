#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <functional>
#include <random>
#include <vector>

#include "mpi/koshkin_n_linear_histogram_stretch/include/ops_mpi.hpp"

std::vector<int> getRandomImageF(int sz) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

TEST(koshkin_n_linear_histogram_stretch_mpi, test_image_imin_imax) {
  boost::mpi::communicator world;

  const int count_size_vector = 9;
  std::vector<int> in_vec = {255, 100, 25, 255, 100, 25, 255, 100, 25};
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataPar->inputs_count.emplace_back(in_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(out_vec_par.size());
  }

  koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out_vec_seq(count_size_vector, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataSeq->inputs_count.emplace_back(in_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_vec_seq.size());

    // Create Task
    koshkin_n_linear_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_vec_par, in_vec);
    ASSERT_EQ(out_vec_seq, in_vec);
    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(koshkin_n_linear_histogram_stretch_mpi, test_correct_image) {
  boost::mpi::communicator world;

  const int width = 150;
  const int height = 150;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = getRandomImageF(count_size_vector);
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataPar->inputs_count.emplace_back(in_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(out_vec_par.size());
  }

  koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out_vec_seq(count_size_vector, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataSeq->inputs_count.emplace_back(in_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_vec_seq.size());

    // Create Task
    koshkin_n_linear_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(koshkin_n_linear_histogram_stretch_mpi, test_large_image) {
  boost::mpi::communicator world;

  const int width = 422;
  const int height = 763;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = getRandomImageF(count_size_vector);
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataPar->inputs_count.emplace_back(in_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(out_vec_par.size());
  }

  koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out_vec_seq(count_size_vector, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataSeq->inputs_count.emplace_back(in_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_vec_seq.size());

    // Create Task
    koshkin_n_linear_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(koshkin_n_linear_histogram_stretch_mpi, test_large_image2) {
  boost::mpi::communicator world;

  const int width = 1024;
  const int height = 512;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = getRandomImageF(count_size_vector);
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataPar->inputs_count.emplace_back(in_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(out_vec_par.size());
  }

  koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out_vec_seq(count_size_vector, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataSeq->inputs_count.emplace_back(in_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_vec_seq.size());

    // Create Task
    koshkin_n_linear_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(koshkin_n_linear_histogram_stretch_mpi, test_large_square_image) {
  boost::mpi::communicator world;

  const int width = 680;
  const int height = 680;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = getRandomImageF(count_size_vector);
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataPar->inputs_count.emplace_back(in_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(out_vec_par.size());
  }

  koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out_vec_seq(count_size_vector, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataSeq->inputs_count.emplace_back(in_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_vec_seq.size());

    // Create Task
    koshkin_n_linear_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(koshkin_n_linear_histogram_stretch_mpi, test_small_square_image) {
  boost::mpi::communicator world;

  const int width = 64;
  const int height = 64;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = getRandomImageF(count_size_vector);
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataPar->inputs_count.emplace_back(in_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(out_vec_par.size());
  }

  koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out_vec_seq(count_size_vector, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataSeq->inputs_count.emplace_back(in_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_vec_seq.size());

    // Create Task
    koshkin_n_linear_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(koshkin_n_linear_histogram_stretch_mpi, test_incorrect_image_size) {
  boost::mpi::communicator world;

  const int count_size_vector = 5;
  std::vector<int> in_vec = {0, 20, 30, 0, 40};
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataPar->inputs_count.emplace_back(in_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(out_vec_par.size());
    koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(koshkin_n_linear_histogram_stretch_mpi, test_incorrect_rgb_image) {
  boost::mpi::communicator world;

  const int width = 3;
  const int height = 3;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = {-2, 20, 30, 0, 355, -25, 10, 260, 255};
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataPar->inputs_count.emplace_back(in_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(out_vec_par.size());
    koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(koshkin_n_linear_histogram_stretch_mpi, test_incorrect_rgb_image2) {
  boost::mpi::communicator world;

  const int width = 2;
  const int height = 3;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = {-2, 20, 30, 0, 355, -25, 10, 260, 255, -2, 20, 30, 22, 33, 44, 72, 89, 90};
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataPar->inputs_count.emplace_back(in_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(out_vec_par.size());
    koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(koshkin_n_linear_histogram_stretch_mpi, test_incorrect_rgb_image3) {
  boost::mpi::communicator world;

  const int width = 70;
  const int height = 15;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = getRandomImageF(count_size_vector);
  in_vec[5] = -25;
  in_vec[17] = 266;
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataPar->inputs_count.emplace_back(in_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(out_vec_par.size());
    koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(koshkin_n_linear_histogram_stretch_mpi, test_incorrect_empty_image) {
  boost::mpi::communicator world;

  const int width = 0;
  const int height = 0;
  const int count_size_vector = width * height * 3;
  std::vector<int> in_vec = {};
  std::vector<int> out_vec_par(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataPar->inputs_count.emplace_back(in_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(out_vec_par.size());
    koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}