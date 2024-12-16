#include "mpi/shuravina_o_jarvis_pass/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>

namespace shuravina_o_jarvis_pass {

void JarvisPassMPI::run() {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int n = points_.size();
  int chunk_size = n / size;
  int start = rank * chunk_size;
  int end = (rank == size - 1) ? n : (rank + 1) * chunk_size;

  std::vector<Point> local_points(points_.begin() + start, points_.begin() + end);

  std::vector<Point> local_hull = jarvis_march(local_points);

  std::vector<Point> global_hull;
  if (rank == 0) {
    global_hull = local_hull;
    for (int i = 1; i < size; i++) {
      std::vector<Point> remote_hull;
      world.recv(i, 0, remote_hull);
      global_hull.insert(global_hull.end(), remote_hull.begin(), remote_hull.end());
    }

    hull_ = jarvis_march(global_hull);
  } else {
    world.send(0, 0, local_hull);
  }
}

std::vector<Point> JarvisPassMPI::get_hull() const { return hull_; }

bool JarvisPassMPI::validation() const { return points_.size() >= 3; }

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