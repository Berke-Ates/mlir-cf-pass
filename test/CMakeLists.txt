configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CF_TEST_DEPENDS FileCheck count not cf-opt)

add_lit_testsuite(
    check-cf-opt "Running the cf regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CF_TEST_DEPENDS}
)

set_target_properties(check-cf-opt PROPERTIES FOLDER "Tests")
add_lit_testsuites(CF-OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${CF_TEST_DEPENDS})
