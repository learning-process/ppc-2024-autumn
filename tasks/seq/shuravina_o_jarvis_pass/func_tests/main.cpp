#include <gtest/gtest.h>

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

TEST(shuravina_o_jarvis_pass_seq, Test_10_Points) {
  const int count_points = 10;
  std::vector<Point> points = getRandomPoints(count_points, 0, 100);
  std::vector<Point> hull;

  JarvisPassSeq jarvis_seq(points);
  ASSERT_TRUE(jarvis_seq.validation());
  jarvis_seq.run();
  hull = jarvis_seq.get_hull();

  std::vector<Point> expected_hull = jarvis_march(points);

  ASSERT_EQ(hull.size(), expected_hull.size());
  for (size_t i = 0; i < hull.size(); i++) {
    EXPECT_EQ(hull[i], expected_hull[i]);
  }
}

TEST(shuravina_o_jarvis_pass_seq, Test_100_Points) {
  const int count_points = 100;
  std::vector<Point> points = getRandomPoints(count_points, 0, 100);
  std::vector<Point> hull;

  JarvisPassSeq jarvis_seq(points);
  ASSERT_TRUE(jarvis_seq.validation());
  jarvis_seq.run();
  hull = jarvis_seq.get_hull();

  std::vector<Point> expected_hull = jarvis_march(points);

  ASSERT_EQ(hull.size(), expected_hull.size());
  for (size_t i = 0; i < hull.size(); i++) {
    EXPECT_EQ(hull[i], expected_hull[i]);
  }
}

TEST(shuravina_o_jarvis_pass_seq, Test_1000_Points) {
  const int count_points = 1000;
  std::vector<Point> points = getRandomPoints(count_points, 0, 100);
  std::vector<Point> hull;

  JarvisPassSeq jarvis_seq(points);
  ASSERT_TRUE(jarvis_seq.validation());
  jarvis_seq.run();
  hull = jarvis_seq.get_hull();

  std::vector<Point> expected_hull = jarvis_march(points);

  ASSERT_EQ(hull.size(), expected_hull.size());
  for (size_t i = 0; i < hull.size(); i++) {
    EXPECT_EQ(hull[i], expected_hull[i]);
  }
}

TEST(shuravina_o_jarvis_pass_seq, Test_Random_Points) {
  const int count_points = 500;
  std::vector<Point> points = getRandomPoints(count_points, 0, 100);
  std::vector<Point> hull;

  JarvisPassSeq jarvis_seq(points);
  ASSERT_TRUE(jarvis_seq.validation());
  jarvis_seq.run();
  hull = jarvis_seq.get_hull();

  std::vector<Point> expected_hull = jarvis_march(points);

  ASSERT_EQ(hull.size(), expected_hull.size());
  for (size_t i = 0; i < hull.size(); i++) {
    EXPECT_EQ(hull[i], expected_hull[i]);
  }
}