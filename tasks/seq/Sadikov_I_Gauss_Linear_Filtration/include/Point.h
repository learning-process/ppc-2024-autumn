#pragma once

#include <iostream>
#include <string>

template <typename T>
class Point {
  T m_red;
  T m_green;
  T m_blue;

 public:
  Point() = default;
  Point(T Red, T Green, T Blue) noexcept : m_red(Red), m_green(Green), m_blue(Blue) {}
  T GetRed() const noexcept { return m_red; }
  T GetGreen() const noexcept { return m_green; }
  T GetBlue() const noexcept { return m_blue; }
  void operator+=(Point value) {
    m_red += value.GetRed();
    m_green += value.GetGreen();
    m_blue += value.GetBlue();
  }
  Point operator*(T value) { return Point(m_red * value, m_green * value, m_blue * value); }
  bool operator!=(Point value) {
    return (static_cast<int>(std::abs(value.GetRed() - m_red)) != 0 or
            static_cast<int>(std::abs(value.GetGreen() - m_green)) != 0 or
            static_cast<int>(std::abs(value.GetBlue() - m_blue)) != 0);
  }
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & m_red;
    ar & m_green;
    ar & m_blue;
  }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, Point<T>& p) {
  return os << p.GetRed() << " " << p.GetGreen() << " " << p.GetBlue() << std::endl;
}
