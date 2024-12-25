#pragma once

#include <algorithm>
#include <iostream>

#define UNUSED(x) (void)(x)

namespace zaitsev_a_jarvis_seq {

template <typename T>

struct Point {
  T x;
  T y;

  Point<T> operator-(const Point<T>& other) { return {x - other.x, y - other.y}; }

  // Vector product
  T operator*(const Point<T>& other) { return x * other.y - other.x * y; }

  bool operator==(const Point<T>& other) const { return x == other.x && y == other.y; }

  friend std::ostream& operator<<(std::ostream& out, Point& p) {
    out << "(" << p.x << "; " << p.y << ")";
    return out;
  }

  bool between(const Point<T>& a, const Point<T>& b) {
    return orientation(a, b, *this) == 0 && x <= std::max(a.x, b.x) && x >= std::min(a.x, b.x) &&
           y <= std::max(a.y, b.y) && y >= std::min(a.y, b.y);
  }

  template <typename Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & x;
    ar & y;
    // UNUSED needed here to supress -Wunused-parameter
    // version arg can't be removed due to requrements of boost for Point struct to be serializable
    UNUSED(version);
  }
};

template <typename T>
int orientation(zaitsev_a_jarvis_seq::Point<T> p, zaitsev_a_jarvis_seq::Point<T> q, zaitsev_a_jarvis_seq::Point<T> r) {
  T vp = (r - q) * (q - p);
  if (vp == 0) return 0;
  return (vp > 0) ? 1 : 2;
}
}  // namespace zaitsev_a_jarvis_seq