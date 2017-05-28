#pragma once
#include <vector>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include "dynet/dynet.h"
#include "dynet/lstm.h"
#include "dynet/expr.h"
#include "utils.h"

class Embedder {
public:
  virtual void NewGraph(ComputationGraph& cg);
  virtual void SetDropout(float rate);
  virtual unsigned Dim() const = 0;
  virtual Expression Embed(const shared_ptr<const Word> word) = 0;
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
};

class StandardEmbedder : public Embedder {
public:
  StandardEmbedder();
  StandardEmbedder(Model& model, unsigned vocab_size, unsigned emb_dim);

  void NewGraph(ComputationGraph& cg) override;
  void SetDropout(float rate) override;
  unsigned Dim() const override;
  Expression Embed(const shared_ptr<const Word> word) override;
private:
  unsigned emb_dim;
  LookupParameter embeddings;
  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<Embedder>(*this);
    ar & emb_dim;
    ar & embeddings;
  }
};
BOOST_CLASS_EXPORT_KEY(StandardEmbedder)
