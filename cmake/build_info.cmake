include_guard(DIRECTORY)

include(timestamp)
include(version)
include(git)

function(FetchBuildInfo)
    FetchTimestamp()
    set(CONFIGURE_TIMESTAMP "${RESULT_VAR}")

    FetchRevHash()
    set(REV_HASH "${RESULT_VAR}")

    FetchVersion()
    set(VERSION "${RESULT_VAR}")

    add_compile_definitions(
        SLICER2_CONFIG_TIMESTAMP=${CONFIGURE_TIMESTAMP}
	    SLICER2_CONFIG_REV=${REV_HASH}
	    SLICER2_VERSION=${VERSION}
    )
endfunction()
