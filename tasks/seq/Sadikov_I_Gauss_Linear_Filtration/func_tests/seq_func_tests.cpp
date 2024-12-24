#include <gtest/gtest.h>
#include <seq/Sadikov_I_Gauss_Linear_Filtration/include/Point.h>
#include <seq/Sadikov_I_Gauss_Linear_Filtration/include/seq_task.h>

TEST(Sadikov_I_Gauss_Linear_Filtration, check_validation) {
  std::vector<Point<double>> in{Point(20.0, 50.0, 60.0), Point(7.0, 8.0, 9.0)};
  std::vector<int> in_index{1, 1};
  std::vector<Point<double>> out(9);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in_index[0]);
  taskData->inputs_count.emplace_back(in_index[1]);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltration task(taskData);
  ASSERT_EQ(task.validation(), false);
}

TEST(Sadikov_I_Gauss_Linear_Filtration, check_validation2) {
  std::vector<Point<double>> in{Point(90.0, 100.0, 110.0), Point(2.0, 3.0, 100.0), Point(0.0, 0.0, 0.0)};
  std::vector<int> in_index{1, 1};
  std::vector<Point<double>> out(2);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in_index[0]);
  taskData->inputs_count.emplace_back(in_index[1]);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltration task(taskData);
  ASSERT_EQ(task.validation(), false);
}

TEST(Sadikov_I_Gauss_Linear_Filtration, check_square_image) {
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
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in_index[0]);
  taskData->inputs_count.emplace_back(in_index[1]);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltration task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  bool flag = true;
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] != squareImageCheck[i]) {
      flag = false;
    }
  }
  ASSERT_EQ(flag, true);
}

TEST(Sadikov_I_Gauss_Linear_Filtration, check_square_image2) {
  std::vector<Point<double>> squareImage2{
      Point(47.0, 128.0, 95.0),  Point(98.0, 134.0, 61.0),  Point(100.0, 100.0, 100.0), Point(45.0, 95.0, 88.0),
      Point(180.0, 90.0, 200.0), Point(56.0, 107.0, 253.0), Point(1.0, 1.0, 200.0),     Point(2.0, 254.0, 128.0),
      Point(0.0, 0.0, 255.0),    Point(46.0, 78.0, 81.0),   Point(255.0, 255.0, 255.0), Point(65.0, 88.0, 100.0),
      Point(190.0, 0.0, 222.0),  Point(102.0, 241.0, 50.0), Point(7.0, 200.0, 190.0),   Point(90.0, 200.0, 255.0)};
  std::vector<Point<double>> squareImage2Check{
      Point(48.0, 61.0, 70.0),  Point(58.0, 75.0, 97.0),   Point(42.0, 76.0, 92.0),   Point(21.0, 63.0, 61.0),
      Point(60.0, 63.0, 126.0), Point(81.0, 95.0, 171.0),  Point(70.0, 118.0, 156.0), Point(40.0, 101.0, 100.0),
      Point(63.0, 46.0, 137.0), Point(88.0, 112.0, 178.0), Point(85.0, 157.0, 174.0), Point(56.0, 120.0, 128.0),
      Point(54.0, 35.0, 89.0),  Point(70.0, 102.0, 109.0), Point(65.0, 139.0, 121.0), Point(46.0, 95.0, 107.0)};
  std::vector<Point<double>> in(std::move(squareImage2));
  std::vector<int> in_index{4, 4};
  std::vector<Point<double>> out(16);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in_index[0]);
  taskData->inputs_count.emplace_back(in_index[1]);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltration task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  bool flag = true;
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] != squareImage2Check[i]) {
      flag = false;
    }
  }
  ASSERT_EQ(flag, true);
}

TEST(Sadikov_I_Gauss_Linear_Filtration, check_rect_image) {
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
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in_index[0]);
  taskData->inputs_count.emplace_back(in_index[1]);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltration task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  bool flag = true;
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] != rectImageCheck[i]) {
      flag = false;
    }
  }
  ASSERT_EQ(flag, true);
}

TEST(Sadikov_I_Gauss_Linear_Filtration, check_rect_image2) {
  std::vector<Point<double>> rectImage2{
      Point(200.0, 31.0, 92.0),  Point(75.0, 198.0, 0.0),   Point(206.0, 51.0, 106.0),  Point(78.0, 66.0, 129.0),
      Point(51.0, 75.0, 100.0),  Point(47.0, 128.0, 95.0),  Point(129.0, 210.0, 223.0), Point(94.0, 126.0, 241.0),
      Point(0.0, 255.0, 255.0),  Point(255.0, 0.0, 0.0),    Point(143.0, 57.0, 68.0),   Point(17.0, 56.0, 41.0),
      Point(82.0, 103.0, 201.0), Point(81.0, 189.0, 255.0), Point(95.0, 41.0, 156.0)};

  std::vector<Point<double>> rectImage2Check{
      Point(63.0, 44.0, 42.0),   Point(81.0, 74.0, 53.0),  Point(61.0, 56.0, 40.0),    Point(75.0, 76.0, 95.0),
      Point(87.0, 120.0, 128.0), Point(54.0, 97.0, 94.0),  Point(93.0, 76.0, 103.0),   Point(89.0, 118.0, 149.0),
      Point(34.0, 100.0, 111.0), Point(109.0, 69.0, 98.0), Point(107.0, 103.0, 143.0), Point(46.0, 78.0, 104.0),
      Point(69.0, 48.0, 77.0),   Point(76.0, 67.0, 107.0), Point(42.0, 42.0, 73.0)};
  std::vector<Point<double>> in(std::move(rectImage2));
  std::vector<int> in_index{5, 3};
  std::vector<Point<double>> out(15);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in_index[0]);
  taskData->inputs_count.emplace_back(in_index[1]);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  Sadikov_I_Gauss_Linear_Filtration::LinearFiltration task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  bool flag = true;
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] != rectImage2Check[i]) {
      flag = false;
    }
  }
  ASSERT_EQ(flag, true);
}
