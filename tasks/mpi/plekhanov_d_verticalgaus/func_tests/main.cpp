#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/plekhanov_d_verticalgaus/include/ops_mpi.hpp"

namespace plekhanov_d_verticalgaus_mpi {

static std::vector<double> getRandomImage(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(0, 255);
  std::vector<double> img(sz);
  for (int i = 0; i < sz; ++i) {
    img[i] = dis(gen);
  }
  return img;
}

void run_test(int heightI, int widthI) {
  boost::mpi::communicator world;

  const int width = widthI;
  const int height = heightI;

  std::vector<double> global_img(width * height, 1);
  std::vector<double> global_ans(width * height, 0);
  std::vector<double> ans(width * height, 1);

  for (int i = 0; i < width; ++i) {
    ans[i] = 0;
    ans[(height - 1) * width + i] = 0;
  }
  for (int i = 1; i < height - 1; ++i) {
    ans[i * width] = 0.75;
    ans[i * width + width - 1] = 0.75;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_img = plekhanov_d_verticalgaus_mpi::getRandomImage(width * height);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataPar->inputs_count.emplace_back(global_img.size());
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  plekhanov_d_verticalgaus_mpi::VerticalGausMPITest testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_ans(width * height, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataSeq->inputs_count.emplace_back(global_img.size());
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    // Create Task
    plekhanov_d_verticalgaus_mpi::VerticalGausSeqTest testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_ans, reference_ans);
  }
}

void validation_test(int heightI, int widthI) {
  boost::mpi::communicator world;

  const int width = widthI;
  const int height = heightI;

  std::vector<double> global_img(width * height, 1);
  std::vector<double> global_ans(width * height, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_img = plekhanov_d_verticalgaus_mpi::getRandomImage(width * height);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataPar->inputs_count.emplace_back(global_img.size());
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  auto taskParallel = std::make_shared<plekhanov_d_verticalgaus_mpi::VerticalGausMPITest>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

}  // namespace plekhanov_d_verticalgaus_mpi

TEST(plekhanov_d_verticalgaus_mpi, validation_zero_zero) { plekhanov_d_verticalgaus_mpi::validation_test(0, 0); }

TEST(plekhanov_d_verticalgaus_mpi, validation_two_two) { plekhanov_d_verticalgaus_mpi::validation_test(2, 2); }

TEST(plekhanov_d_verticalgaus_mpi, validation_three_two) { plekhanov_d_verticalgaus_mpi::validation_test(3, 2); }

TEST(plekhanov_d_verticalgaus_mpi, validation_two_three) { plekhanov_d_verticalgaus_mpi::validation_test(2, 3); }

TEST(plekhanov_d_verticalgaus_mpi, validation_hundred_two) { plekhanov_d_verticalgaus_mpi::validation_test(100, 2); }

TEST(plekhanov_d_verticalgaus_mpi, three_four) { plekhanov_d_verticalgaus_mpi::run_test(3, 4); }

TEST(plekhanov_d_verticalgaus_mpi, four_five) { plekhanov_d_verticalgaus_mpi::run_test(4, 5); }

TEST(plekhanov_d_verticalgaus_mpi, five_six) { plekhanov_d_verticalgaus_mpi::run_test(5, 6); }

TEST(plekhanov_d_verticalgaus_mpi, six_eight) { plekhanov_d_verticalgaus_mpi::run_test(6, 8); }

TEST(plekhanov_d_verticalgaus_mpi, seven_nine) { plekhanov_d_verticalgaus_mpi::run_test(7, 9); }

TEST(plekhanov_d_verticalgaus_mpi, ten_twenty) { plekhanov_d_verticalgaus_mpi::run_test(10, 20); }

TEST(plekhanov_d_verticalgaus_mpi, twenty_twenty) { plekhanov_d_verticalgaus_mpi::run_test(20, 20); }

TEST(plekhanov_d_verticalgaus_mpi, twenty_thirty) { plekhanov_d_verticalgaus_mpi::run_test(20, 30); }

TEST(plekhanov_d_verticalgaus_mpi, thirty_forty) { plekhanov_d_verticalgaus_mpi::run_test(30, 40); }

TEST(plekhanov_d_verticalgaus_mpi, forty_fifty) { plekhanov_d_verticalgaus_mpi::run_test(40, 50); }

TEST(plekhanov_d_verticalgaus_mpi, fifty_sixty) { plekhanov_d_verticalgaus_mpi::run_test(50, 60); }

TEST(plekhanov_d_verticalgaus_mpi, sixty_seventy) { plekhanov_d_verticalgaus_mpi::run_test(60, 70); }

TEST(plekhanov_d_verticalgaus_mpi, seventy_eighty) { plekhanov_d_verticalgaus_mpi::run_test(70, 80); }

TEST(plekhanov_d_verticalgaus_mpi, eighty_ninety) { plekhanov_d_verticalgaus_mpi::run_test(80, 90); }

TEST(plekhanov_d_verticalgaus_mpi, ninety_hundred) { plekhanov_d_verticalgaus_mpi::run_test(90, 100); }

TEST(plekhanov_d_verticalgaus_mpi, fifty_fifty) { plekhanov_d_verticalgaus_mpi::run_test(50, 50); }

TEST(plekhanov_d_verticalgaus_mpi, hundred_hundred) { plekhanov_d_verticalgaus_mpi::run_test(100, 100); }

TEST(plekhanov_d_verticalgaus_mpi, hundred_hundredtwenty) { plekhanov_d_verticalgaus_mpi::run_test(100, 120); }

TEST(plekhanov_d_verticalgaus_mpi, hundredtwenty_hundred) { plekhanov_d_verticalgaus_mpi::run_test(120, 200); }

TEST(plekhanov_d_verticalgaus_mpi, hundredufifty_twohundred) { plekhanov_d_verticalgaus_mpi::run_test(150, 200); }