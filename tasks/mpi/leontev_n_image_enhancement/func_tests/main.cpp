#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <functional>
#include <random>
#include <vector>

#include "mpi/leontev_n_image_enhancement/include/ops_mpi.hpp"

namespace leontev_n_image_enhancement_mpi {
std::vector<int> getRandomImage(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
}  // namespace leontev_n_image_enhancement_mpi

inline void taskEmplacement(std::shared_ptr<ppc::core::TaskData> &taskDataPar, std::vector<int> &global_vec,
                            std::vector<int> &global_sum) {
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataPar->inputs_count.emplace_back(global_vec.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
  taskDataPar->outputs_count.emplace_back(global_sum.size());
}

TEST(leontev_n_image_enhancement_mpi, test_image_imin_imax) {
  boost::mpi::communicator world;

  const int vector_size = 6;
  std::vector<int> in_vec = {255, 0, 255, 255, 0, 255};
  std::vector<int> out_vec_par(vector_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskEmplacement(taskDataPar, in_vec, out_vec_par);
  }

  leontev_n_image_enhancement_mpi::MPIImgEnhancementParallel MPIImgEnhancementParallel(taskDataPar);
  ASSERT_EQ(MPIImgEnhancementParallel.validation(), true);
  MPIImgEnhancementParallel.pre_processing();
  MPIImgEnhancementParallel.run();
  MPIImgEnhancementParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out_vec_seq(vector_size, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskEmplacement(taskDataSeq, in_vec, out_vec_seq);

    // Create Task
    leontev_n_image_enhancement_mpi::MPIImgEnhancementSequential MPIImgEnhancementSequential(taskDataSeq);
    ASSERT_EQ(MPIImgEnhancementSequential.validation(), true);
    MPIImgEnhancementSequential.pre_processing();
    MPIImgEnhancementSequential.run();
    MPIImgEnhancementSequential.post_processing();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(leontev_n_image_enhancement_mpi, test_image_1) {
  boost::mpi::communicator world;

  const int width = 300;
  const int height = 300;
  const int vector_size = width * height * 3;
  std::vector<int> in_vec = leontev_n_image_enhancement_mpi::getRandomImage(vector_size);
  std::vector<int> out_vec_par(vector_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskEmplacement(taskDataPar, in_vec, out_vec_par);
  }

  leontev_n_image_enhancement_mpi::MPIImgEnhancementParallel MPIImgEnhancementParallel(taskDataPar);
  ASSERT_EQ(MPIImgEnhancementParallel.validation(), true);
  MPIImgEnhancementParallel.pre_processing();
  MPIImgEnhancementParallel.run();
  MPIImgEnhancementParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out_vec_seq(vector_size, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskEmplacement(taskDataSeq, in_vec, out_vec_seq);

    // Create Task
    leontev_n_image_enhancement_mpi::MPIImgEnhancementSequential MPIImgEnhancementSequential(taskDataSeq);
    ASSERT_EQ(MPIImgEnhancementSequential.validation(), true);
    MPIImgEnhancementSequential.pre_processing();
    MPIImgEnhancementSequential.run();
    MPIImgEnhancementSequential.post_processing();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(leontev_n_image_enhancement_mpi, test_image_2) {
  boost::mpi::communicator world;

  const int width = 600;
  const int height = 600;
  const int vector_size = width * height * 3;
  std::vector<int> in_vec = leontev_n_image_enhancement_mpi::getRandomImage(vector_size);
  std::vector<int> out_vec_par(vector_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskEmplacement(taskDataPar, in_vec, out_vec_par);
  }

  leontev_n_image_enhancement_mpi::MPIImgEnhancementParallel MPIImgEnhancementParallel(taskDataPar);
  ASSERT_EQ(MPIImgEnhancementParallel.validation(), true);
  MPIImgEnhancementParallel.pre_processing();
  MPIImgEnhancementParallel.run();
  MPIImgEnhancementParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out_vec_seq(vector_size, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskEmplacement(taskDataSeq, in_vec, out_vec_seq);

    // Create Task
    leontev_n_image_enhancement_mpi::MPIImgEnhancementSequential MPIImgEnhancementSequential(taskDataSeq);
    ASSERT_EQ(MPIImgEnhancementSequential.validation(), true);
    MPIImgEnhancementSequential.pre_processing();
    MPIImgEnhancementSequential.run();
    MPIImgEnhancementSequential.post_processing();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(leontev_n_image_enhancement_mpi, test_incorrect_image_size) {
  boost::mpi::communicator world;

  const int vector_size = 5;
  std::vector<int> in_vec = {0, 10, 10, 10, 60};
  std::vector<int> out_vec_par(vector_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskEmplacement(taskDataPar, in_vec, out_vec_par);
    leontev_n_image_enhancement_mpi::MPIImgEnhancementParallel MPIImgEnhancementParallel(taskDataPar);
    ASSERT_EQ(MPIImgEnhancementParallel.validation(), false);
  }
}

TEST(leontev_n_image_enhancement_mpi, test_incorrect_rgb_image) {
  boost::mpi::communicator world;

  const int width = 4;
  const int height = 4;
  const int vector_size = width * height * 3;
  std::vector<int> in_vec = leontev_n_image_enhancement_mpi::getRandomImage(vector_size);
  in_vec[0] = -100;
  std::vector<int> out_vec_par(vector_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskEmplacement(taskDataPar, in_vec, out_vec_par);
    leontev_n_image_enhancement_mpi::MPIImgEnhancementParallel MPIImgEnhancementParallel(taskDataPar);
    ASSERT_EQ(MPIImgEnhancementParallel.validation(), false);
  }
}

TEST(leontev_n_image_enhancement_mpi, test_incorrect_rgb_image2) {
  boost::mpi::communicator world;

  const int width = 2;
  const int height = 3;
  const int vector_size = width * height * 3;
  std::vector<int> in_vec = {-2, -10, -20, 0, 555, -25, 50, 265, 0, 0, 0, 0, -22, 0, 1, 4, 105, 90};
  std::vector<int> out_vec_par(vector_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskEmplacement(taskDataPar, in_vec, out_vec_par);
    leontev_n_image_enhancement_mpi::MPIImgEnhancementParallel MPIImgEnhancementParallel(taskDataPar);
    ASSERT_EQ(MPIImgEnhancementParallel.validation(), false);
  }
}

TEST(leontev_n_image_enhancement_mpi, test_incorrect_empty_image) {
  boost::mpi::communicator world;

  const int width = 0;
  const int height = 0;
  const int vector_size = width * height * 3;
  std::vector<int> in_vec = {};
  std::vector<int> out_vec_par(vector_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskEmplacement(taskDataPar, in_vec, out_vec_par);
    leontev_n_image_enhancement_mpi::MPIImgEnhancementParallel MPIImgEnhancementParallel(taskDataPar);
    ASSERT_EQ(MPIImgEnhancementParallel.validation(), false);
  }
}
