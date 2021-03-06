/**
 * @file simple_residue_termination.hpp
 * @author Sumedh Ghaisas
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_METHODS_AMF_UPDATE_RULES_SVD_BATCHLEARNING_HPP
#define __MLPACK_METHODS_AMF_UPDATE_RULES_SVD_BATCHLEARNING_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

/**
 * This class implements SVD batch learning with momentum. This procedure is
 * described in the paper 'A Guide to singular Value Decomposition'
 * by Chih-Chao Ma. Class implements 'Algorithm 4' given in the paper.
 * This factorizer decomposes the matrix V into two matrices W and H such that
 * sum of sum of squared error between V and W*H is minimum. This optimization is
 * performed with gradient descent. To make gradient descent faster momentum is
 * added.
 */
class SVDBatchLearning
{
 public:
  /**
   * SVD Batch learning constructor.
   *
   * @param u step value used in batch learning
   * @param kw regularization constant for W matrix
   * @param kh regularization constant for H matrix
   * @param momentum momentum applied to batch learning process
   */
  SVDBatchLearning(double u = 0.0002,
                   double kw = 0,
                   double kh = 0,
                   double momentum = 0.9,
                   double min = -DBL_MIN,
                   double max = DBL_MAX)
        : u(u), kw(kw), kh(kh), min(min), max(max), momentum(momentum)
    {}

  template<typename MatType>
  void Initialize(const MatType& dataset, const size_t rank)
  {
    const size_t n = dataset.n_rows;
    const size_t m = dataset.n_cols;

    mW.zeros(n, rank);
    mH.zeros(rank, m);
  }

  /**
   * The update rule for the basis matrix W.
   * The function takes in all the matrices and only changes the
   * value of the W matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix to be updated.
   * @param H Encoding matrix.
   */
  template<typename MatType>
  inline void WUpdate(const MatType& V,
                      arma::mat& W,
                      const arma::mat& H)
  {
    size_t n = V.n_rows;
    size_t m = V.n_cols;

    size_t r = W.n_cols;

    mW = momentum * mW;

    arma::mat deltaW(n, r);
    deltaW.zeros();

    for(size_t i = 0;i < n;i++)
    {
      for(size_t j = 0;j < m;j++)
      {
        double val;
        if((val = V(i, j)) != 0)
          deltaW.row(i) += (val - arma::dot(W.row(i), H.col(j))) *
                                                  arma::trans(H.col(j));
      }
      if(kw != 0) deltaW.row(i) -= kw * W.row(i);
    }

    mW += u * deltaW;
    W += mW;
  }

  /**
   * The update rule for the encoding matrix H.
   * The function takes in all the matrices and only changes the
   * value of the H matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix.
   * @param H Encoding matrix to be updated.
   */
  template<typename MatType>
  inline void HUpdate(const MatType& V,
                      const arma::mat& W,
                      arma::mat& H)
  {
    size_t n = V.n_rows;
    size_t m = V.n_cols;

    size_t r = W.n_cols;

    mH = momentum * mH;

    arma::mat deltaH(r, m);
    deltaH.zeros();

    for(size_t j = 0;j < m;j++)
    {
      for(size_t i = 0;i < n;i++)
      {
        double val;
        if((val = V(i, j)) != 0)
          deltaH.col(j) += (val - arma::dot(W.row(i), H.col(j))) *
                                                    arma::trans(W.row(i));
      }
      if(kh != 0) deltaH.col(j) -= kh * H.col(j);
    }

    mH += u*deltaH;
    H += mH;
  }

 private:
  double u;
  double kw;
  double kh;
  double min;
  double max;
  double momentum;

  arma::mat mW;
  arma::mat mH;
};

//! TODO : Merge this template specialized function for sparse matrix using
//!        common row_col_iterator

/**
 * WUpdate function specialization for sparse matrix
 */
template<>
inline void SVDBatchLearning::WUpdate<arma::sp_mat>(const arma::sp_mat& V,
                                                    arma::mat& W,
                                                    const arma::mat& H)
{
  size_t n = V.n_rows;

  size_t r = W.n_cols;

  mW = momentum * mW;

  arma::mat deltaW(n, r);
  deltaW.zeros();

  for(arma::sp_mat::const_iterator it = V.begin();it != V.end();it++)
  {
    size_t row = it.row();
    size_t col = it.col();
    deltaW.row(it.row()) += (*it - arma::dot(W.row(row), H.col(col))) *
                                                  arma::trans(H.col(col));
  }

  if(kw != 0) for(size_t i = 0; i < n; i++)
  {
    deltaW.row(i) -= kw * W.row(i);
  }

  mW += u * deltaW;
  W += mW;
}

template<>
inline void SVDBatchLearning::HUpdate<arma::sp_mat>(const arma::sp_mat& V,
                                                    const arma::mat& W,
                                                    arma::mat& H)
{
  size_t m = V.n_cols;

  size_t r = W.n_cols;

  mH = momentum * mH;

  arma::mat deltaH(r, m);
  deltaH.zeros();

  for(arma::sp_mat::const_iterator it = V.begin();it != V.end();it++)
  {
    size_t row = it.row();
    size_t col = it.col();
    deltaH.col(col) += (*it - arma::dot(W.row(row), H.col(col))) *
                                                arma::trans(W.row(row));
  }

  if(kh != 0) for(size_t j = 0; j < m; j++)
  {
    deltaH.col(j) -= kh * H.col(j);
  }

  mH += u*deltaH;
  H += mH;
}

} // namespace amf
} // namespace mlpack

#endif


