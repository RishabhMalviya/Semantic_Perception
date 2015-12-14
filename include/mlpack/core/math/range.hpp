/**
 * @file range.hpp
 *
 * Definition of the Range class, which represents a simple range with a lower
 * and upper bound.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_CORE_MATH_RANGE_HPP
#define __MLPACK_CORE_MATH_RANGE_HPP

namespace mlpack {
namespace math {

/**
 * Simple real-valued range.  It contains an upper and lower bound.
 */
class Range
{
 private:
  double lo; /// The lower bound.
  double hi; /// The upper bound.

 public:
  /** Initialize to an empty set (where lo > hi). */
  inline Range();

  /***
   * Initialize a range to enclose only the given point (lo = point, hi =
   * point).
   *
   * @param point Point that this range will enclose.
   */
  inline Range(const double point);

  /**
   * Initializes to specified range.
   *
   * @param lo Lower bound of the range.
   * @param hi Upper bound of the range.
   */
  inline Range(const double lo, const double hi);

  //! Get the lower bound.
  inline double Lo() const { return lo; }
  //! Modify the lower bound.
  inline double& Lo() { return lo; }

  //! Get the upper bound.
  inline double Hi() const { return hi; }
  //! Modify the upper bound.
  inline double& Hi() { return hi; }

  /**
   * Gets the span of the range (hi - lo).
   */
  inline double Width() const;

  /**
   * Gets the midpoint of this range.
   */
  inline double Mid() const;

  /**
   * Expands this range to include another range.
   *
   * @param rhs Range to include.
   */
  inline Range& operator|=(const Range& rhs);

  /**
   * Expands this range to include another range.
   *
   * @param rhs Range to include.
   */
  inline Range operator|(const Range& rhs) const;

  /**
   * Shrinks this range to be the overlap with another range; this makes an
   * empty set if there is no overlap.
   *
   * @param rhs Other range.
   */
  inline Range& operator&=(const Range& rhs);

  /**
   * Shrinks this range to be the overlap with another range; this makes an
   * empty set if there is no overlap.
   *
   * @param rhs Other range.
   */
  inline Range operator&(const Range& rhs) const;

  /**
   * Scale the bounds by the given double.
   *
   * @param d Scaling factor.
   */
  inline Range& operator*=(const double d);

  /**
   * Scale the bounds by the given double.
   *
   * @param d Scaling factor.
   */
  inline Range operator*(const double d) const;

  /**
   * Scale the bounds by the given double.
   *
   * @param d Scaling factor.
   */
  friend inline Range operator*(const double d, const Range& r); // Symmetric.

  /**
   * Compare with another range for strict equality.
   *
   * @param rhs Other range.
   */
  inline bool operator==(const Range& rhs) const;

  /**
   * Compare with another range for strict equality.
   *
   * @param rhs Other range.
   */
  inline bool operator!=(const Range& rhs) const;

  /**
   * Compare with another range.  For Range objects x and y, x < y means that x
   * is strictly less than y and does not overlap at all.
   *
   * @param rhs Other range.
   */
  inline bool operator<(const Range& rhs) const;

  /**
   * Compare with another range.  For Range objects x and y, x < y means that x
   * is strictly less than y and does not overlap at all.
   *
   * @param rhs Other range.
   */
  inline bool operator>(const Range& rhs) const;

  /**
   * Determines if a point is contained within the range.
   *
   * @param d Point to check.
   */
  inline bool Contains(const double d) const;

  /**
   * Determines if another range overlaps with this one.
   *
   * @param r Other range.
   *
   * @return true if ranges overlap at all.
   */
  inline bool Contains(const Range& r) const;

  /**
   * Returns a string representation of an object.
   */
  inline std::string ToString() const;

};

}; // namespace math
}; // namespace mlpack

// Include inlined implementation.
#include "range_impl.hpp"

#endif // __MLPACK_CORE_MATH_RANGE_HPP