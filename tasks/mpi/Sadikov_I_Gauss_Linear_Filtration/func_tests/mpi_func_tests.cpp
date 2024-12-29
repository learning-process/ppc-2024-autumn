#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>

#include "mpi/Sadikov_I_Gauss_Linear_Filtration/include/ops_mpi.h"

namespace Sadikov_I_Gauss_Linear_Filtration {
std::vector<Point<double>> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<Point<double>> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] =
        Point(static_cast<double>(gen() % 255), static_cast<double>(gen() % 255), static_cast<double>(gen() % 255));
  }
  return vec;
}
}  // namespace Sadikov_I_Gauss_Linear_Filtration

TEST(Sadikov_I_Gauss_Linear_Filtration, check_validation) {
  boost::mpi::communicator world;
  std::vector<Point<double>> in{Point(15.0, 20.0, 30.0), Point(10.0, 20.0, 100.0), Point(100.0, 128.0, 200.0)};
  std::vector<int> in_index{3, 1};
  std::vector<Point<double>> out(3);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in_index[0]);
  taskData->inputs_count.emplace_back(in_index[1]);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI task(taskData);
  if (world.rank() == 0) {
    ASSERT_EQ(task.validation(), false);
  }
}

TEST(Sadikov_I_Gauss_Linear_Filtration, check_validation2) {
  boost::mpi::communicator world;
  std::vector<Point<double>> in{Point(75.0, 200.0, 30.0), Point(10.0, 20.0, 100.0),  Point(100.0, 128.0, 200.0),
                                Point(75.0, 20.0, 30.0),  Point(10.0, 20.0, 100.0),  Point(100.0, 128.0, 200.0),
                                Point(75.0, 20.0, 30.0),  Point(100.0, 20.0, 100.0), Point(100.0, 128.0, 200.0)};
  std::vector<int> in_index{3, 3};
  std::vector<Point<double>> out(10);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in_index[0]);
  taskData->inputs_count.emplace_back(in_index[1]);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI task(taskData);
  if (world.rank() == 0) {
    ASSERT_EQ(task.validation(), false);
  }
}

TEST(Sadikov_I_Gauss_Linear_Filtration, check_square_image) {
  boost::mpi::communicator world;
  std::vector<Point<double>> squareImage{
      Point(44.0, 2.0, 1.0),      Point(56.0, 11.0, 23.0),   Point(51.0, 75.0, 100.0),
      Point(100.0, 245.0, 140.0), Point(98.0, 134.0, 61.0),  Point(100.0, 100.0, 100.0),
      Point(19.0, 200.0, 98.0),   Point(44.0, 128.0, 128.0), Point(67.0, 198.0, 11.0)};
  std::vector<Point<double>> squareImageCheck{
      Point(35.0, 42.0, 24.0),  Point(50.0, 54.0, 42.0),  Point(37.0, 39.0, 40.0),
      Point(47.0, 102.0, 59.0), Point(70.0, 122.0, 76.0), Point(54.0, 81.0, 53.0),
      Point(29.0, 97.0, 57.0),  Point(46.0, 117.0, 65.0), Point(38.0, 78.0, 35.0)};

  std::vector<Point<double>> in(std::move(squareImage));
  std::vector<int> in_index{3, 3};
  std::vector<Point<double>> out(9);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  if (world.rank() == 0) {
    bool flag = true;
    for (size_t i = 0; i < out.size(); ++i) {
      if (out[i] != squareImageCheck[i]) {
        flag = false;
      }
    }
    ASSERT_EQ(flag, true);
  }
}

TEST(Sadikov_I_Gauss_Linear_Filtration, check_square_image2) {
  boost::mpi::communicator world;
  std::vector<Point<double>> squareImage2{
      Point(47.0, 128.0, 95.0),  Point(98.0, 134.0, 61.0),  Point(100.0, 100.0, 100.0), Point(45.0, 95.0, 88.0),
      Point(180.0, 90.0, 200.0), Point(56.0, 107.0, 253.0), Point(1.0, 1.0, 200.0),     Point(2.0, 254.0, 128.0),
      Point(0.0, 0.0, 255.0),    Point(46.0, 78.0, 81.0),   Point(255.0, 255.0, 255.0), Point(65.0, 88.0, 100.0),
      Point(190.0, 0.0, 222.0),  Point(102.0, 241.0, 50.0), Point(7.0, 200.0, 190.0),   Point(90.0, 200.0, 255.0)};
  std::vector<Point<double>> in(std::move(squareImage2));
  std::vector<int> in_index{4, 4};
  std::vector<Point<double>> out(16);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  if (world.rank() == 0) {
    std::vector<Point<double>> outSeq(91);
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in_index[0]);
    taskDataSeq->inputs_count.emplace_back(in_index[1]);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outSeq.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
    Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationSeq taskSeq(taskDataSeq);
    taskSeq.validation();
    taskSeq.pre_processing();
    taskSeq.run();
    taskSeq.post_processing();
    bool flag = true;
    for (size_t i = 0; i < out.size(); ++i) {
      if (out[i] != outSeq[i]) {
        flag = false;
      }
    }
    ASSERT_EQ(flag, true);
  }
}

TEST(Sadikov_I_Gauss_Linear_Filtration, check_rect_image) {
  boost::mpi::communicator world;
  std::vector<Point<double>> rectImage{Point(140.0, 68.0, 69.0),   Point(79.0, 171.0, 10.0), Point(200.0, 245.0, 187.0),
                                       Point(255.0, 255.0, 255.0), Point(123.0, 77.0, 92.0), Point(87.0, 204.0, 99.0),
                                       Point(100.0, 211.0, 19.0),  Point(177.0, 62.0, 14.0), Point(47.0, 128.0, 95.0),
                                       Point(129.0, 210.0, 223.0), Point(199.0, 43.0, 4.0),  Point(0.0, 0.0, 0.0)};
  std::vector<Point<double>> rectImageCheck{
      Point(60.0, 59.0, 34.0), Point(85.0, 120.0, 54.0),  Point(114.0, 148.0, 81.0), Point(106.0, 105., 78.0),
      Point(74.0, 93.0, 68.0), Point(115.0, 160.0, 89.0), Point(137.0, 159.0, 78.0), Point(110.0, 92.0, 51.0),
      Point(47.0, 77.0, 65.0), Point(84.0, 110.0, 78.0),  Point(88.0, 80.0, 39.0),   Point(54.0, 28.0, 3.0)};
  std::vector<Point<double>> in(std::move(rectImage));
  std::vector<int> in_index{3, 4};
  std::vector<Point<double>> out(12);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  if (world.rank() == 0) {
    bool flag = true;
    for (size_t i = 0; i < out.size(); ++i) {
      if (out[i] != rectImageCheck[i]) {
        flag = false;
      }
    }
    ASSERT_EQ(flag, true);
  }
}

TEST(Sadikov_I_Gauss_Linear_Filtration, check_rect_image2) {
  boost::mpi::communicator world;
  std::vector<Point<double>> rectImage2{
      Point(200.0, 31.0, 92.0),  Point(75.0, 198.0, 0.0),   Point(206.0, 51.0, 106.0),  Point(78.0, 66.0, 129.0),
      Point(51.0, 75.0, 100.0),  Point(47.0, 128.0, 95.0),  Point(129.0, 210.0, 223.0), Point(94.0, 126.0, 241.0),
      Point(0.0, 255.0, 255.0),  Point(255.0, 0.0, 0.0),    Point(143.0, 57.0, 68.0),   Point(17.0, 56.0, 41.0),
      Point(82.0, 103.0, 201.0), Point(81.0, 189.0, 255.0), Point(95.0, 41.0, 156.0)};
  std::vector<Point<double>> in(std::move(rectImage2));
  std::vector<int> in_index{5, 3};
  std::vector<Point<double>> out(15);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  if (world.rank() == 0) {
    std::vector<Point<double>> outSeq(91);
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in_index[0]);
    taskDataSeq->inputs_count.emplace_back(in_index[1]);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outSeq.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
    Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationSeq taskSeq(taskDataSeq);
    taskSeq.validation();
    taskSeq.pre_processing();
    taskSeq.run();
    taskSeq.post_processing();
    bool flag = true;
    for (size_t i = 0; i < out.size(); ++i) {
      if (out[i] != outSeq[i]) {
        flag = false;
      }
    }
    ASSERT_EQ(flag, true);
  }
}

TEST(Sadikov_I_Gauss_Linear_Filtration, check_rect_image3) {
  boost::mpi::communicator world;
  std::vector<Point<double>> rectImage3{
      Point(200.0, 31.0, 92.0),   Point(75.0, 198.0, 0.0),    Point(206.0, 51.0, 106.0),  Point(78.0, 66.0, 129.0),
      Point(51.0, 75.0, 100.0),   Point(47.0, 128.0, 95.0),   Point(129.0, 210.0, 223.0), Point(94.0, 126.0, 241.0),
      Point(82.0, 103.0, 201.0),  Point(81.0, 189.0, 255.0),  Point(95.0, 41.0, 156.0),   Point(0.0, 255.0, 255.0),
      Point(255.0, 0.0, 0.0),     Point(143.0, 57.0, 68.0),   Point(17.0, 56.0, 41.0),    Point(245.0, 128.0, 98.0),
      Point(75.0, 1.0, 20.0),     Point(144.0, 100.0, 200.0), Point(210.0, 54.0, 122.0),  Point(0.0, 0.0, 100.0),
      Point(98.0, 211.0, 73.0),   Point(35.0, 62.0, 199.0),   Point(81.0, 122.0, 210.0),  Point(229.0, 119.0, 45.0),
      Point(51.0, 75.0, 100.0),   Point(47.0, 128.0, 95.0),   Point(129.0, 210.0, 223.0), Point(94.0, 126.0, 241.0),
      Point(200.0, 31.0, 92.0),   Point(75.0, 198.0, 0.0),    Point(206.0, 51.0, 106.0),  Point(78.0, 66.0, 129.0),
      Point(75.0, 1.0, 20.0),     Point(144.0, 100.0, 200.0), Point(210.0, 54.0, 122.0),  Point(0.0, 0.0, 100.0),
      Point(100.0, 100.0, 100.0), Point(92.0, 122.0, 289.0),  Point(2.0, 49.0, 100.0),    Point(240.0, 55.0, 129.0),
      Point(222.0, 100.0, 0.0),   Point(32.0, 33.0, 199.0),   Point(251.0, 112.0, 52.0),  Point(44.0, 61.0, 166.0),
      Point(47.0, 128.0, 95.0),   Point(51.0, 75.0, 100.0),   Point(129.0, 210.0, 223.0), Point(50.0, 50.0, 50.0),
      Point(199.0, 2.0, 4.0),     Point(94.0, 126.0, 241.0),  Point(0.0, 0.0, 221.0),     Point(45.0, 124.0, 10.0),
      Point(229.0, 0.0, 21.0),    Point(31.0, 69.0, 100.0),   Point(98.0, 67.0, 124.0),   Point(128.0, 128.0, 128.0),
      Point(200.0, 191.0, 10.0),  Point(41.0, 75.0, 129.0),   Point(48.0, 1.0, 221.0),    Point(59.0, 225.0, 231.0),
      Point(87.0, 63.0, 187.0),   Point(37.0, 7.0, 100.0),    Point(46.0, 84., 35.0),     Point(0.0, 128.0, 255.0),
      Point(24.0, 51.0, 221.0),   Point(123.0, 79.0, 12.0),   Point(175.0, 0.0, 0.0),     Point(45.0, 98.0, 125.0),
      Point(75.0, 1.0, 20.0),     Point(144.0, 100.0, 200.0), Point(210.0, 54.0, 122.0),  Point(0.0, 0.0, 100.0),
      Point(200.0, 31.0, 92.0),   Point(75.0, 198.0, 0.0),    Point(206.0, 51.0, 106.0),  Point(78.0, 66.0, 129.0),
      Point(98.0, 211.0, 73.0),   Point(35.0, 62.0, 199.0),   Point(81.0, 122.0, 210.0),  Point(229.0, 119.0, 45.0),
      Point(229.0, 0.0, 21.0),    Point(31.0, 69.0, 100.0),   Point(98.0, 67.0, 124.0),   Point(128.0, 128.0, 128.0),
      Point(82.0, 103.0, 201.0),  Point(81.0, 189.0, 255.0),  Point(95.0, 41.0, 156.0),   Point(0.0, 255.0, 255.0),
      Point(192.0, 234.0, 88.0),  Point(24.0, 122.0, 87.0),   Point(45.0, 29.0, 112.0)};
  std::vector<Point<double>> in(std::move(rectImage3));
  std::vector<int> in_index{7, 13};
  std::vector<Point<double>> out(91);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  if (world.rank() == 0) {
    std::vector<Point<double>> outSeq(91);
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in_index[0]);
    taskDataSeq->inputs_count.emplace_back(in_index[1]);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outSeq.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
    Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationSeq taskSeq(taskDataSeq);
    taskSeq.validation();
    taskSeq.pre_processing();
    taskSeq.run();
    taskSeq.post_processing();
    bool flag = true;
    for (size_t i = 0; i < out.size(); ++i) {
      if (out[i] != outSeq[i]) {
        flag = false;
      }
    }
    ASSERT_EQ(flag, true);
  }
}

TEST(Sadikov_I_Gauss_Linear_Filtration, check_empty_image) {
  boost::mpi::communicator world;
  std::vector<Point<double>> emptyImage;
  std::vector<Point<double>> emptyImageCheck;
  std::vector<Point<double>> in(std::move(emptyImage));
  std::vector<int> in_index{0, 0};
  std::vector<Point<double>> out(0);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
}

TEST(Sadikov_I_Gauss_Linear_Filtration, check_random_image) {
  boost::mpi::communicator world;
  std::vector<Point<double>> in = Sadikov_I_Gauss_Linear_Filtration::getRandomVector(64);
  std::vector<int> in_index{8, 8};
  std::vector<Point<double>> out(64);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  if (world.rank() == 0) {
    std::vector<Point<double>> outSeq(64);
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in_index[0]);
    taskDataSeq->inputs_count.emplace_back(in_index[1]);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outSeq.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
    Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationSeq taskSeq(taskDataSeq);
    taskSeq.validation();
    taskSeq.pre_processing();
    taskSeq.run();
    taskSeq.post_processing();
    bool flag = true;
    std::cout << std::endl;
    for (size_t i = 0; i < out.size(); ++i) {
      if (out[i] != outSeq[i]) {
        flag = false;
      }
    }
    ASSERT_EQ(flag, true);
  }
}

TEST(Sadikov_I_Gauss_Linear_Filtration, check_random_image2) {
  boost::mpi::communicator world;
  std::vector<Point<double>> in = Sadikov_I_Gauss_Linear_Filtration::getRandomVector(99);
  std::vector<int> in_index{9, 11};
  std::vector<Point<double>> out(99);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationMPI task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  if (world.rank() == 0) {
    std::vector<Point<double>> outSeq(99);
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in_index[0]);
    taskDataSeq->inputs_count.emplace_back(in_index[1]);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outSeq.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
    Sadikov_I_Gauss_Linear_Filtration::LinearFiltrationSeq taskSeq(taskDataSeq);
    taskSeq.validation();
    taskSeq.pre_processing();
    taskSeq.run();
    taskSeq.post_processing();
    bool flag = true;
    for (size_t i = 0; i < out.size(); ++i) {
      if (out[i] != outSeq[i]) {
        flag = false;
      }
    }
    ASSERT_EQ(flag, true);
  }
}