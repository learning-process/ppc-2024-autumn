#pragma once

#include <mpi.h>

#include <vector>

namespace shuravina_o_jarvis_pass {

struct Point {
  int x, y;
  Point(int x = 0, int y = 0) : x(x), y(y) {}

  bool operator<(const Point& p) const { return (x < p.x) || (x == p.x && y < p.y); }
  bool operator==(const Point& p) const { return x == p.x && y == p.y; }
};

class JarvisPassMPI {
 public:
  JarvisPassMPI(std::vector<Point>& points) : points_(points) {}
  void run();
  std::vector<Point> get_hull() const;

  bool validation() const;

 private:
  std::vector<Point>& points_;
  std::vector<Point> hull_;
};

std::vector<Point> jarvis_march(const std::vector<Point>& points);

}  // namespace shuravina_o_jarvis_pass