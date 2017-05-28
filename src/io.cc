#include <fstream>
#include "io.h"

void Serialize(Dict& vocab, const DependencyOutputModel& model, Model& dynet_model, const Trainer* const trainer) {
  int r = ftruncate(fileno(stdout), 0);
  if (r != 0) {}
  fseek(stdout, 0, SEEK_SET);

  boost::archive::binary_oarchive oa(cout);
  oa & dynet_model;
  oa & vocab;
  oa & model;
  oa & trainer;
}

void Deserialize(const string& filename, Dict& vocab, DependencyOutputModel& model, Model& dynet_model, Trainer*& trainer) {
  ifstream f(filename);
  boost::archive::binary_iarchive ia(f);
  ia & dynet_model;
  ia & vocab;
  ia & model;
  ia & trainer;
  f.close();
}
