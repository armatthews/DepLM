#pragma once
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <vector>
#include <functional>
#include "dynet/dict.h"
#include "dynet/training.h"
#include "deplm.h"
#include "utils.h"

using namespace std;
using namespace dynet;


void Serialize(Dict& vocab, const DependencyOutputModel& model, Model& dynet_model, const Trainer* const trainer);
void Deserialize(const string& filename, Dict& vocab, DependencyOutputModel& model, Model& dynet_model, Trainer*& trainer);
