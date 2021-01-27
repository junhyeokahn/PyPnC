add_library(my_util SHARED "util.cpp"
                           "mlp_model.cpp")
target_link_libraries(my_util my_yaml)
