#include "seq/shuravina_o_jarvis_pass/include/ops_seq.hpp"

namespace shuravina_o_jarvis_pass {

void JarvisPassSeq::run() { hull_ = jarvis_march(points_); }

std::vector<Point> JarvisPassSeq::get_hull() const { return hull_; }

bool JarvisPassSeq::validation() const { return points_.size() >= 3; }

std::vector<Point> jarvis_march(const std::vector<Point>& points) {
  int n = points.size();
  if (n < 3) return points;

  std::vector<Point> hull;

  int l = 0;
  for (int i = 1; i < n; i++) {
    if (points[i] < points[l]) {
      l = i;
    }
  }

  int p = l;
  int q;
  do {
    hull.push_back(points[p]);
    q = (p + 1) % n;

    for (int i = 0; i < n; i++) {
      if ((points[i].y - points[p].y) * (points[q].x - points[i].x) -
              (points[i].x - points[p].x) * (points[q].y - points[i].y) <
          0) {
        q = i;
      }
    }

    p = q;
  } while (p != l);

  return hull;
}

}  // namespace shuravina_o_jarvis_pass