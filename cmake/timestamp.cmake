include_guard(DIRECTORY)

function(FetchTimestamp)
    string(TIMESTAMP CONFIGURE_TIMESTAMP "%s")
    message(STATUS "Configure-time timestamp: ${CONFIGURE_TIMESTAMP}")

    set(RESULT_VAR "${CONFIGURE_TIMESTAMP}" PARENT_SCOPE)
endfunction()
