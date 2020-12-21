#ifndef YAML_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define YAML_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include "yaml/include/myYaml/parser.h"
#include "yaml/include/myYaml/emitter.h"
#include "yaml/include/myYaml/emitterstyle.h"
#include "yaml/include/myYaml/stlemitter.h"
#include "yaml/include/myYaml/exceptions.h"

#include "yaml/include/myYaml/node/node.h"
#include "yaml/include/myYaml/node/impl.h"
#include "yaml/include/myYaml/node/convert.h"
#include "yaml/include/myYaml/node/iterator.h"
#include "yaml/include/myYaml/node/detail/impl.h"
#include "yaml/include/myYaml/node/parse.h"
#include "yaml/include/myYaml/node/emit.h"
#include "yaml/include/myYaml/yaml_eigen.h"

#endif  // YAML_H_62B23520_7C8E_11DE_8A39_0800200C9A66
