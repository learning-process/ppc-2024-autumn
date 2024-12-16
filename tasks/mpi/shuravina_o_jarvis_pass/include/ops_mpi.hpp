#pragma once

#include <boost/mpi/communicator.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

namespace shuravina_o_jarvis_pass {

struct Point {
  int x, y;
  Point(int x = 0, int y = 0) : x(x), y(y) {}

  bool operator<(const Point& p) const { return (x < p.x) || (x == p.x && y < p.y); }

  bool operator==(const Point& p) const { return x == p.x && y == p.y; }

 private:
  friend class boost::serialization::access;

  template <typename Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & x;
    ar & y;
  }
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