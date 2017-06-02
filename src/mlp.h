#pragma once
#include <boost/serialization/access.hpp>
#include "dynet/dynet.h"
#include "dynet/expr.h"

using namespace std;
using namespace dynet;
using namespace dynet::expr;

class MLP {
public:
  MLP();
  MLP(Model& model, unsigned input_size, unsigned hidden_size, unsigned output_size);
  void NewGraph(ComputationGraph& cg);
  void SetDropout(float rate);
  Expression Feed(Expression input) const;

private:
  float dropout_rate;

  Parameter p_wIH;
  Parameter p_wHO;
  Parameter p_wHb;
  Parameter p_wOb;

  Expression wIH;
  Expression wHO;
  Expression wHb;
  Expression wOb;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & p_wIH;
    ar & p_wHO;
    ar & p_wHb;
    ar & p_wOb;
  }
};
