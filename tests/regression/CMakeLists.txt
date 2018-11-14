set(sourceList
    "access_sycl_cl_types.cpp"
    "anatomy_sycl_app_parallel_for.cpp"
    "anatomy_sycl_app_single_task.cpp"
    "example_sycl_app.cpp"
    "functors_nd_range_kernels.cpp"
    "naive_square_matrix_rotation.cpp"
    "random_number_generation.cpp"
    "reduction_sum.cpp"
    "reduction_sum_local.cpp"
    "simple_vector_addition.cpp"
    "vectors_in_kernel.cpp"
    "work_efficient_prefix_sum.cpp")

add_test_group("regression" "${sourceList}")
