file(GLOB_RECURSE sources "*.cpp")
file(GLOB yaml_headers "include/myYaml/*.h")
file(GLOB yaml_contrib_headers "include/myYaml/contrib/*.h")
file(GLOB yaml_node_headers "include/myYaml/node/*.h")
file(GLOB yaml_node_detail_headers "include/myYaml/node/detail/*.h")

add_library(my_yaml SHARED ${sources}
                           ${yaml_headers}
                           ${yaml_contrib_headers}
                           ${yaml_node_headers}
                           ${yaml_node_detail_headers})
