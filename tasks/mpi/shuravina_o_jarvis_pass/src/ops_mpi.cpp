#include "mpi/shuravina_o_jarvis_pass/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>

namespace shuravina_o_jarvis_pass {

void JarvisPassMPI::run() {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = points_.size();
  int chunk_size = n / size;
  int start = rank * chunk_size;
  int end = (rank == size - 1) ? n : (rank + 1) * chunk_size;

  std::vector<Point> local_points(points_.begin() + start, points_.begin() + end);
  std::vector<Point> local_hull = jarvis_march(local_points);

  if (rank == 0) {
    hull_ = local_hull;
    for (int i = 1; i < size; i++) {
      int count;
      MPI_Recv(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::vector<Point> remote_hull(count);
      MPI_Recv(remote_hull.data(), count * sizeof(Point), MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      hull_.insert(hull_.end(), remote_hull.begin(), remote_hull.end());
    }
    hull_ = jarvis_march(hull_);
  } else {
    int count = local_hull.size();
    MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(local_hull.data(), count * sizeof(Point), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
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