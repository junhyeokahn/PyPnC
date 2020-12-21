#include "graphbuilderadapter.h"

#include "yaml/include/myYaml/parser.h"  // IWYU pragma: keep

namespace YAML {
class GraphBuilderInterface;

void* BuildGraphOfNextDocument(Parser& parser,
                               GraphBuilderInterface& graphBuilder) {
  GraphBuilderAdapter eventHandler(graphBuilder);
  if (parser.HandleNextDocument(eventHandler)) {
    return eventHandler.RootNode();
  } else {
    return NULL;
  }
}
}
