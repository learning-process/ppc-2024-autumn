#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "seq/shuravina_o_jarvis_pass/include/ops_seq.hpp"

using namespace shuravina_o_jarvis_pass;

static std::vector<Point> getRandomPoints(int count, int min_coord, int max_coord) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(min_coord, max_coord);
  std::vector<Point> points(count);
  for (int i = 0; i < count; i++) {
    points[i] = Point(dist(gen), dist(gen));
  }
  return points;
}

TEST(shuravina_o_jarvis_pass_seq_perf, Test_10000_Points) {
  const int count_points = 10000;
  std::vector<Point> points = getRandomPoints(count_points, 0, 100);
  std::vector<Point> hull;

  JarvisPassSeq jarvis_seq(points);
  ASSERT_TRUE(jarvis_seq.validation());

  auto start = std::chrono::high_resolution_clock::now();
  jarvis_seq.run();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken for 10000 points (Sequential): " << elapsed.count() << " seconds" << std::endl;
}