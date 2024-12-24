
#include <limits>
#include <random>
#include <stack>
#include <thread>

#include "seq/solovyev_d_convex_hull_binary_image_components/include/header.hpp"

namespace solovyev_d_convex_hull_binary_image_components_seq {

double Point::relativeAngle(Point other) const { return std::atan2((other.y * -1) - (y * -1), other.x - x); }

Image::Image() {
  image = std::vector<Point>{};
  sizeX = 0;
  sizeY = 0;
}

Image::Image(std::vector<int> data, int dimX, int dimY) {
  for (size_t i = 0; i < data.size(); i++) {
    int pointY = i / dimX;
    int pointX = i - pointY * dimX;
    Point point = {pointX, pointY, data[i]};
    image.push_back(point);
  }
  sizeX = dimX;
  sizeY = dimY;
}

Point Image::getPoint(int x, int y) {
  if (x >= 0 && x <= sizeX && y >= 0 && y <= sizeY) {
    return image[y * sizeX + x];
  }
  Point point = {x, y, 0};
  return point;
}

void Image::setPoint(int x, int y, int value) { image[y * sizeX + x].value = value; }

std::vector<int> Image::getComponents() { return components; }

int Image::newComponent() {
  int id = components.back() + 1;
  components.push_back(id);
  return id;
}

void Image::clearComponents() {
  components.clear();
  components.push_back(0);
}

void Image::removeComponent(int n) {
  auto index = find(components.begin(), components.end(), n);
  if (index != components.end()) {
    components.erase(index);
  }
}

void Image::fixComponents() {
  for (size_t i = 0; i < components.size(); i++) {
    components[i] = i;
  }
}

static int crossProduct(Point a, Point b, Point c) {
  return (b.x - a.x) * (-1 * c.y - -1 * a.y) - (-1 * b.y - -1 * a.y) * (c.x - a.x);
}

static std::vector<Point> convexHull(std::vector<Point> component) {
  std::sort(component.begin(), component.end(), [](const Point &L, const Point &R) { return L.y < R.y; });
  Point point = component.back();
  std::sort(component.begin(), component.end(), [&](const Point &L, const Point &R) {
    if (point.relativeAngle(L) == point.relativeAngle(R)) {
      double distanceL = (L.x - point.x) * (L.x - point.x) + (L.y - point.y) * (L.y - point.y);
      double distanceR = (R.x - point.x) * (R.x - point.x) + (R.y - point.y) * (R.y - point.y);
      return distanceL < distanceR;
    }
    return point.relativeAngle(L) < point.relativeAngle(R);
  });
  std::stack<Point> hull;
  hull.push(component[0]);
  if (component.size() != 1) {
    hull.push(component[1]);
  }
  for (size_t i = 2; i < component.size(); ++i) {
    while (hull.size() > 1) {
      Point top = hull.top();
      hull.pop();
      Point nextToTop = hull.top();
      if (crossProduct(nextToTop, top, component[i]) > 0) {
        hull.push(top);
        break;
      }
    }
    hull.push(component[i]);
  }

  std::vector<Point> result;
  while (!hull.empty()) {
    result.push_back(hull.top());
    hull.pop();
  }
  std::reverse(result.begin(), result.end());
  return result;
}

bool ConvexHullBinaryImageComponentsSequential::pre_processing() {
  internal_order_test();

  // Init data vector
  int *input_ = reinterpret_cast<int *>(taskData->inputs[0]);
  int dimX = *reinterpret_cast<int *>(taskData->inputs[1]);
  int dimY = *reinterpret_cast<int *>(taskData->inputs[2]);
  std::vector<int> data(input_, input_ + taskData->inputs_count[0]);
  image = Image(data, dimX, dimY);
  return true;
}

bool ConvexHullBinaryImageComponentsSequential::validation() {
  internal_order_test();
  // Check count elements of output
  bool isSizeCorrect =
      ((*reinterpret_cast<int *>(taskData->inputs[1])) * (*reinterpret_cast<int *>(taskData->inputs[2])) ==
       (int)(taskData->inputs_count[0]));
  return ((*reinterpret_cast<int *>(taskData->inputs[1]) >= 0 && *reinterpret_cast<int *>(taskData->inputs[2]) >= 0) &&
          isSizeCorrect);
}

bool ConvexHullBinaryImageComponentsSequential::run() {
  internal_order_test();
  image.clearComponents();
  results.clear();
  equivalenceTable.clear();
  // First phase
  for (int y = 0; y < image.sizeY; y++) {
    for (int x = 0; x < image.sizeX; x++) {
      Point point = image.getPoint(x, y);
      Point diag = image.getPoint(x - 1, y - 1);
      Point left = image.getPoint(x - 1, y);
      Point up = image.getPoint(x, y - 1);
      if (point.value != 0) {
        if (diag.value != 0) {
          image.setPoint(x, y, diag.value);
        } else {
          if (left.value == 0 && up.value == 0) {
            image.setPoint(x, y, image.newComponent());
          } else if (left.value == 0 && up.value != 0) {
            image.setPoint(x, y, up.value);
          } else if (left.value != 0 && up.value == 0) {
            image.setPoint(x, y, left.value);
          } else {
            image.setPoint(x, y, up.value);
            if (up.value != left.value) {
              equivalenceTable.push_back(eqUnit{left.value, up.value});
            }
          }
        }
      }
    }
  }

  // Second phase
  std::sort(equivalenceTable.begin(), equivalenceTable.end(),
            [&](const eqUnit &L, const eqUnit &R) { return L.replaceable > R.replaceable; });
  for (int y = 0; y < image.sizeY; y++) {
    for (int x = 0; x < image.sizeX; x++) {
      for (size_t i = 0; i < equivalenceTable.size(); i++) {
        if (image.getPoint(x, y).value == equivalenceTable[i].replaceable) {
          image.setPoint(x, y, equivalenceTable[i].replacement);
          image.removeComponent(equivalenceTable[i].replaceable);
        }
      }
    }
  }

  // Third phase
  for (int y = 0; y < image.sizeY; y++) {
    for (int x = 0; x < image.sizeX; x++) {
      for (size_t i = 0; i < image.getComponents().size(); i++) {
        if (image.getPoint(x, y).value == image.getComponents()[i]) {
          image.setPoint(x, y, i);
        }
      }
    }
  }
  image.fixComponents();
  // Getting hulls for every component
  for (size_t j = 1; j < image.getComponents().size(); j++) {
    std::vector<Point> component;
    for (int y = 0; y < image.sizeY; y++) {
      for (int x = 0; x < image.sizeX; x++) {
        if (image.getPoint(x, y).value == (int)j) {
          component.push_back(image.getPoint(x, y));
        }
      }
    }
    std::vector<int> result = linearizePoints(convexHull(component));
    results.push_back(result);
  }
  return true;
}

bool ConvexHullBinaryImageComponentsSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < results.size(); i++) {
    for (size_t j = 0; j < results[i].size(); j++) {
      reinterpret_cast<int *>(taskData->outputs[i])[j] = results[i][j];
    }
  }
  return true;
}
}  // namespace solovyev_d_convex_hull_binary_image_components_seq