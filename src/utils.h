#pragma once
#include <vector>
#include <map>
#include <string>
#include <tuple>
#include <memory>
/*#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>*/
#include "dynet/dict.h"
#include "dynet/expr.h"

using namespace std;
using namespace dynet;
using namespace dynet::expr;

typedef int WordId;

struct Word {
  virtual ~Word();
};

struct StandardWord : public Word {
  explicit StandardWord(WordId id);
  WordId id;
};

typedef vector<shared_ptr<Word>> OutputSentence;

unsigned Sample(const vector<float>& dist);

unsigned int UTF8Len(unsigned char x);
unsigned int UTF8StringLen(const string& x);

vector<string> tokenize(string input, string delimiter, unsigned max_times);
vector<string> tokenize(string input, string delimiter);
vector<string> tokenize(string input, char delimiter);

string strip(const string& input);
vector<string> strip(const vector<string>& input, bool removeEmpty = false);

float logsumexp(const vector<float>& v);
vector<Expression> MakeGRUInitialState(Expression c, unsigned gru_dim, unsigned gru_layer_count);
vector<Expression> MakeLSTMInitialState(Expression c, unsigned lstm_dim, unsigned lstm_layer_count);
string vec2str(Expression expr);
bool same_value(Expression e1, Expression e2);

vector<OutputSentence> ReadText(const string& filename, Dict& vocab);
