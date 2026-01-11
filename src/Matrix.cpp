/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Matrix.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////
MatrixFloatView fromRawBuffer(float *pBuffer, Index iRows, Index iCols) {
  return MatrixFloat::from_raw_buffer(pBuffer, iRows, iCols);
}
///////////////////////////////////////////////////////////////////////////
const MatrixFloatView fromRawBuffer(const float *pBuffer, Index iRows,
                                    Index iCols) {
  return MatrixFloat::from_raw_buffer((float *)pBuffer, iRows, iCols);
}
///////////////////////////////////////////////////////////////////////////
void copyInto(const MatrixFloat &mToCopy, MatrixFloat &m, Index iStartRow) {
  assert(m.cols() == mToCopy.cols());
  assert(iStartRow + mToCopy.rows() <= m.rows());

  memcpy(m.data() + iStartRow * m.cols(), mToCopy.data(),
         mToCopy.size() * sizeof(float));
}
///////////////////////////////////////////////////////////////////////////
MatrixFloatView createView(MatrixFloat &mRef) {
  return fromRawBuffer(mRef.data(), mRef.rows(), mRef.cols());
}
///////////////////////////////////////////////////////////////////////////
const MatrixFloatView viewResize(const MatrixFloat &m, Index iRows,
                                 Index iCols) {
  assert(m.size() >= iRows * iCols);
  return fromRawBuffer(m.data(), iRows, iCols);
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat rowWiseSum(const MatrixFloat &m) {
  MatrixFloat out(m.rows(), 1);

  for (Index i = 0; i < m.rows(); i++) {
    float fSum = 0;
    for (Index j = 0; j < m.cols(); j++)
      fSum += m(i, j);

    out(i) = fSum;
  }
  return out;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat rowWiseSumSq(const MatrixFloat &m) {
  MatrixFloat out(m.rows(), 1);

  for (Index i = 0; i < m.rows(); i++) {
    float fSum = 0;
    for (Index j = 0; j < m.cols(); j++)
      fSum += m(i, j) * m(i, j);

    out(i) = fSum;
  }
  return out;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat rowWiseAdd(const MatrixFloat &m, const MatrixFloat &d) {
  assert(m.rows() == d.rows());
  assert(d.cols() == 1);

  MatrixFloat out = m;

  for (Index i = 0; i < m.rows(); i++) {
    float fVal = d(i);
    for (Index j = 0; j < m.cols(); j++)
      out(i, j) += fVal;
  }
  return out;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat rowWiseMult(const MatrixFloat &m, const MatrixFloat &d) {
  assert(m.rows() == d.rows());
  assert(d.cols() == 1);

  MatrixFloat out = m;

  for (Index i = 0; i < m.rows(); i++) {
    float fVal = d(i);
    for (Index j = 0; j < m.cols(); j++)
      out(i, j) *= fVal;
  }
  return out;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat addRowVector(const MatrixFloat &m, const MatrixFloat &v) {
  assert(v.rows() == 1);
  assert(v.cols() == m.cols());

  MatrixFloat out = m;

  for (Index i = 0; i < m.rows(); i++) {
    for (Index j = 0; j < m.cols(); j++)
      out(i, j) += v(0, j);
  }
  return out;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat rowWiseDivide(const MatrixFloat &m, const MatrixFloat &d) {
  assert(m.rows() == d.rows());
  assert(d.cols() == 1);

  MatrixFloat out = m;

  for (Index i = 0; i < m.rows(); i++) {
    float fVal = d(i);
    for (Index j = 0; j < m.cols(); j++)
      out(i, j) /= fVal;
  }
  return out;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat colWiseSum(const MatrixFloat &m) {
  MatrixFloat out(1, m.cols());
  out.setZero();

  for (Index i = 0; i < m.rows(); i++) {
    for (Index j = 0; j < m.cols(); j++)
      out(j) += m(i, j);
  }
  return out;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat colWiseSumSq(const MatrixFloat &m) {
  MatrixFloat out(1, m.cols());
  out.setZero();

  for (Index i = 0; i < m.rows(); i++) {
    for (Index j = 0; j < m.cols(); j++)
      out(j) += m(i, j) * m(i, j);
  }
  return out;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat colWiseMean(const MatrixFloat &m) {
  return colWiseSum(m) / float(m.rows());
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat colWiseMin(const MatrixFloat &m) {
  // todo check empty m
  MatrixFloat out(1, m.cols());

  for (Index j = 0; j < m.cols(); j++)
    out(j) = m(0, j);

  for (Index i = 1; i < m.rows(); i++) {
    for (Index j = 0; j < m.cols(); j++)
      out(j) = std::min(out(j), m(i, j));
  }
  return out;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat colWiseMax(const MatrixFloat &m) {
  // todo check empty m
  MatrixFloat out(1, m.cols());

  for (Index j = 0; j < m.cols(); j++)
    out(j) = m(0, j);

  for (Index i = 1; i < m.rows(); i++) {
    for (Index j = 0; j < m.cols(); j++)
      out(j) = std::max(out(j), m(i, j));
  }
  return out;
}
///////////////////////////////////////////////////////////////////////////
std::vector<Index> randPerm(Index iSize) {
  std::vector<Index> v(iSize);
  for (Index i = 0; i < iSize; i++)
    v[i] = i;

  std::shuffle(v.begin(), v.end(), randomEngine());

  return v;
}
///////////////////////////////////////////////////////////////////////////
void applyRowPermutation(const std::vector<Index> &vPermutation,
                         const MatrixFloat &mIn, MatrixFloat &mPermuted) {
  assert(vPermutation.size() == mIn.rows());
  mPermuted.resizeLike(mIn);

  for (Index i = 0; i < mIn.rows(); i++) {
    // todo optimize
    for (Index j = 0; j < mIn.cols(); j++)
      mPermuted(i, j) = mIn(vPermutation[i], j);
  }
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat decimate(const MatrixFloat &m, Index iRatio) {
  MatrixFloat out(m.rows() / iRatio, m.cols());

  for (Index i = 0; i < out.rows(); i++) {
    for (Index j = 0; j < out.cols(); j++)
      out(i, j) = m(i * iRatio, j);
  }
  return out;
}
///////////////////////////////////////////////////////////////////////////
Index argmax(const MatrixFloat &m) {
  Index iBest = 0;
  float fMax = m(0);
  for (Index i = 1; i < m.size(); i++)
    if (m(i) > fMax) {
      fMax = m(i);
      iBest = i;
    }
  return iBest;
}
///////////////////////////////////////////////////////////////////////////
void rowsArgmax(const MatrixFloat &m, MatrixFloat &argM) {
  argM.resize(m.rows(), 1);

  for (Index i = 0; i < m.rows(); i++) {
    Index iBest = 0;
    float fMax = m(i, 0);
    for (Index j = 1; j < m.cols(); j++)
      if (m(i, j) > fMax) {
        fMax = m(i, j);
        iBest = j;
      }
    argM(i) = (float)iBest;
  }
}
///////////////////////////////////////////////////////////////////////////
void clamp(MatrixFloat &m, float fClampMin, float fClampMax) {
  for (Index i = 0; i < m.size(); i++) {
    float v = m(i);
    if (v > fClampMax)
      m(i) = fClampMax;
    else if (v < fClampMin)
      m(i) = fClampMin;
  }
}
///////////////////////////////////////////////////////////////////////////
string toString(const MatrixFloat &m) {
  stringstream ss;
  for (Index iL = 0; iL < m.rows(); iL++) {
    for (Index iR = 0; iR < m.cols(); iR++)
      ss << m(iL, iR) << " ";
    if (iL + 1 < m.rows())
      ss << endl;
  }

  return ss.str();
}
///////////////////////////////////////////////////////////////////////////
const MatrixFloat fromFile(const string &sFile) {
  vector<float> vf;
  std::fstream f(sFile, std::ios::in);
  Index iNbLine = 0;
  while (!f.eof() && (!f.bad()) && (!f.fail())) {
    string s;
    getline(f, s);

    if (s.empty())
      continue;

    std::replace(s.begin(), s.end(), ',',
                 ' '); // replace ',' by spaces if present

    iNbLine++;

    std::stringstream ss;
    ss.str(s);
    while (!ss.eof()) {
      float sF;
      ss >> sF;
      vf.push_back(sF);
    }
  }

  if (iNbLine == 0)
    return MatrixFloat();

  MatrixFloat r(iNbLine, (Index)vf.size() / iNbLine); // todo check size
  std::copy(vf.begin(), vf.end(), r.data());
  return r;
}
///////////////////////////////////////////////////////////////////////////
const MatrixFloat fromString(const string &s) {
  MatrixFloat r;
  vector<float> vf;
  std::stringstream ss(s);
  Index iNbCols = 0, iNbLine = 0;

  while (!ss.eof()) {
    float sF;
    ss >> sF;
    vf.push_back(sF);
  }

  iNbCols = (Index)std::count(s.begin(), s.end(), '\n') + 1;
  iNbLine = (Index)vf.size() / iNbCols;

  r.resize(iNbLine, iNbCols);
  std::copy(vf.begin(), vf.end(), r.data());
  return r;
}
///////////////////////////////////////////////////////////////////////////
bool toFile(const string &sFile, const MatrixFloat &m) {
  std::fstream f(sFile, std::ios::out);
  for (Index iL = 0; iL < m.rows(); iL++) {
    for (Index iR = 0; iR < m.cols(); iR++)
      f << m(iL, iR) << " ";
    f << endl;
  }

  return true;
}
///////////////////////////////////////////////////////////////////////////
// create a row view starting at iStartRow ending at iEndRow (not included)
const MatrixFloatView viewRow(const MatrixFloat &m, Index iStartRow,
                              Index iEndRow) {
  assert(iStartRow < iEndRow); // iEndRow not included
  assert(m.rows() >= iEndRow);

  return fromRawBuffer(m.data() + iStartRow * m.cols(), iEndRow - iStartRow,
                       (Index)m.cols());
}
///////////////////////////////////////////////////////////////////////////
const MatrixFloat colExtract(const MatrixFloat &m, Index iStartCol,
                             Index iEndCol) {
  // TODO slow for now, use view on strided matrix later
  MatrixFloat mr(m.rows(), iEndCol - iStartCol);

  for (Index iR = 0; iR < mr.rows(); iR++)
    for (Index iC = 0; iC < mr.cols(); iC++)
      mr(iR, iC) = m(iR, iC + iStartCol);

  return mr;
}
///////////////////////////////////////////////////////////////////////////
std::default_random_engine &randomEngine() {
  static std::default_random_engine rng;
  return rng;
}
///////////////////////////////////////////////////////////////////////////
void setRandomUniform(MatrixFloat &m, float fMin, float fMax) {
  std::uniform_real_distribution<float> dis(fMin, fMax);

  for (Index i = 0; i < m.size(); i++)
    m(i) = dis(randomEngine());
}
///////////////////////////////////////////////////////////////////////////
void setRandomNormal(MatrixFloat &m, float fMean, float fNormal) {
  std::normal_distribution<float> dis(fMean, fNormal);

  for (Index i = 0; i < m.size(); i++)
    m(i) = dis(randomEngine());
}
///////////////////////////////////////////////////////////////////////////
void setQuickBernoulli(MatrixFloat &m, float fProba) {
  // quick bernoulli ; resolution proba = 1/65536.
  // speed is 6x faster than bernoulli_distribution !
  unsigned int uiLimit = int(fProba * 65536.);
  for (Index i = 0; i < m.size(); i++)
    m(i) = (randomEngine()() & 0xffff) < uiLimit; // quick, precise enough

  /*
          bernoulli_distribution dis(fProba);
          for(Index i=0;i<m.size();i++)
                  m(i)=(float)(dis(randomEngine())); //slow
  */
}
///////////////////////////////////////////////////////////////////////////
void channelWiseAdd(MatrixFloat &mIn, Index iNbSamples, Index iNbChannels,
                    Index iNbRows, Index iNbCols, const MatrixFloat &weight) {
  assert(weight.size() == iNbChannels);
  assert(mIn.rows() == iNbSamples);
  assert(mIn.size() == iNbSamples * iNbChannels * iNbRows * iNbCols);

  // todo optimize a lot
  for (Index iS = 0; iS < iNbSamples; iS++)
    for (Index iH = 0; iH < iNbChannels; iH++)
      for (Index iR = 0; iR < iNbRows; iR++)
        for (Index iC = 0; iC < iNbCols; iC++)
          mIn(iS, iH * iNbRows * iNbCols + iR * iNbCols + iC) += weight(iH);
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat channelWiseMean(const MatrixFloat &m, Index iNbSamples,
                            Index iNbChannels, Index iNbRows, Index iNbCols) {
  assert(m.rows() == iNbSamples);
  assert(m.size() == iNbSamples * iNbChannels * iNbRows * iNbCols);

  MatrixFloat mMean;
  mMean.setZero(1, iNbChannels);

  // todo optimize a lot
  for (Index iS = 0; iS < iNbSamples; iS++)
    for (Index iH = 0; iH < iNbChannels; iH++)
      for (Index iR = 0; iR < iNbRows; iR++)
        for (Index iC = 0; iC < iNbCols; iC++)
          mMean(0, iH) += m(iS, iH * iNbRows * iNbCols + iR * iNbCols + iC);

  mMean *= (1.f / iNbSamples * iNbRows * iNbCols);

  return mMean;
}
///////////////////////////////////////////////////////////////////////////
float *rowPtr(MatrixFloat &m, Index iRow) { return m.data() + m.cols() * iRow; }
///////////////////////////////////////////////////////////////////////////
const float *rowPtr(const MatrixFloat &m, Index iRow) {
  return m.data() + m.cols() * iRow;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat tanh(const MatrixFloat &m) {
  MatrixFloat r = m;
  for (Index i = 0; i < r.size(); i++)
    r(i) = ::tanh(m(i));

  return r;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat oneMinusSquare(const MatrixFloat &m) {
  MatrixFloat r = m;
  for (Index i = 0; i < r.size(); i++)
    r(i) = 1.f - m(i) * m(i);

  return r;
}
///////////////////////////////////////////////////////////////////////////
void reverseData(float *pData, Index iSize) {
  float *pDataEnd = pData + iSize - 1;
  Index iHalfSize = iSize >> 1;
  for (Index i = 0; i <= iHalfSize; i++)
    *pData++ = *pDataEnd--;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat concatenateRows(MatrixFloat a, MatrixFloat b) {
  assert(a.cols() == b.cols());
  MatrixFloat out(a.rows() + b.rows(), a.cols());
  for (int i = 0; i < a.rows(); i++) {
    out.row(i) = a.row(i);
  }
  for (int i = 0; i < b.rows(); i++) {
    out.row(a.rows() + i) = b.row(i);
  }
  return out;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat concatenateCols(MatrixFloat a, MatrixFloat b) {
  assert(a.rows() == b.rows());
  MatrixFloat out(a.rows(), a.cols() + b.cols());
  for (Index r = 0; r < a.rows(); r++) {
    for (Index c = 0; c < a.cols(); c++) {
      out(r, c) = a(r, c);
    }
    for (Index c = 0; c < b.cols(); c++) {
      out(r, a.cols() + c) = b(r, c);
    }
  }
  return out;
}
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

std::string bytesToHex(const char* data, size_t length) {
    std::ostringstream oss;
    for (size_t i = 0; i < length; ++i) {
        unsigned char byte = static_cast<unsigned char>(data[i]);
        // Manually format each byte as two hex digits
        char high = (byte >> 4) & 0x0F;
        char low = byte & 0x0F;

        // Convert to hex characters
        auto toHexChar = [](char nibble) -> char {
            return nibble < 10 ? ('0' + nibble) : ('A' + nibble - 10);
            };

        oss << toHexChar(high) << toHexChar(low);
    }
    return oss.str();
}

std::vector<char> hexToBytes(const std::string& hex) {
    std::vector<char> bytes;
    bytes.reserve(hex.length() / 2);

    for (size_t i = 0; i < hex.length(); i += 2) {
        char high = hex[i];
        char low = hex[i + 1];

        // Convert each hex digit to its numeric value
        auto hexCharToInt = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            return 0;
            };

        // Combine high and low nibbles
        unsigned char byte = (hexCharToInt(high) << 4) | hexCharToInt(low);
        bytes.push_back(static_cast<char>(byte));
    }
    return bytes;
}

void saveMatrix(std::ostream& os, const MatrixFloat& m) {
    Index rows = m.rows();
    Index cols = m.cols();

    // Convert dimensions to hex
    std::string rowsHex = bytesToHex(reinterpret_cast<const char*>(&rows), sizeof(Index));
    std::string colsHex = bytesToHex(reinterpret_cast<const char*>(&cols), sizeof(Index));

    // Write dimensions as hex strings with delimiters
    os << rowsHex << "\n" << colsHex << "\n";

    // Convert and write matrix data if not empty
    if (rows > 0 && cols > 0) {
        std::string dataHex = bytesToHex(
            reinterpret_cast<const char*>(m.data()),
            rows * cols * sizeof(float)
        );
        os << dataHex << "\n";
    }
}

void loadMatrix(std::istream& is, MatrixFloat& m) {
    std::string rowsHex, colsHex;

    // Read dimension hex strings
    is >> std::ws;
    std::getline(is, rowsHex);

    is >> std::ws;
    std::getline(is, colsHex);

    // Convert hex back to binary
    auto rowsBytes = hexToBytes(rowsHex);
    auto colsBytes = hexToBytes(colsHex);

    Index rows = *reinterpret_cast<const Index*>(rowsBytes.data());
    Index cols = *reinterpret_cast<const Index*>(colsBytes.data());

    m.resize(rows, cols);

    // Read matrix data if not empty
    if (rows > 0 && cols > 0) {
        std::string dataHex;
        std::getline(is, dataHex);

        auto dataBytes = hexToBytes(dataHex);
        std::memcpy(m.data(), dataBytes.data(), rows * cols * sizeof(float));
    }
}
///////////////////////////////////////////////////////////////////////////
} // namespace beednn