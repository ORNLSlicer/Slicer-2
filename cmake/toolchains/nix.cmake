if(NOT "$ENV{NIX_PROFILE_LOADED}")
    message(STATUS "Setting up environment via Nix...")

    find_program(NIX_EXECUTABLE nix)

    if(NOT NIX_EXECUTABLE)
        message(FATAL_ERROR "Nix executable not found, is it installed?")
    endif()

    execute_process(
        COMMAND ${NIX_EXECUTABLE} print-dev-env ${NIX_DEVELOP_PROFILE} -L --json
        RESULT_VARIABLE NIX_DEVELOP_RESULT
        OUTPUT_VARIABLE NIX_DEVELOP_OUTPUT
    )

    if(${NIX_DEVELOP_RESULT})
        message(FATAL_ERROR "Failed to setup environment, check above for details.")
    endif()

    message(STATUS "Loading environment...")

    function(LoadNixPathVar)
        string(JSON VAR_VALUE ERROR_VARIABLE JSON_ERROR GET "${NIX_DEVELOP_OUTPUT}" "variables" "PATH" "value")
        if("${JSON_ERROR}" STREQUAL "NOTFOUND")
            message(STATUS "Appending PATH")
            string(APPEND ENV{PATH} ":${VAR_VALUE}")
        endif()
    endfunction()

    function(LoadNixEnvVar LOAD_VAR)
        string(JSON VAR_VALUE ERROR_VARIABLE JSON_ERROR GET "${NIX_DEVELOP_OUTPUT}" "variables" "${LOAD_VAR}" "value")
        if("${JSON_ERROR}" STREQUAL "NOTFOUND")
            message(STATUS "Setting ${LOAD_VAR}")
            set(ENV{${LOAD_VAR}} ${VAR_VALUE})
        endif()
    endfunction()

    function(LoadAllEnvVars)
        list(
            APPEND EXCLUDE_VARS
            "buildPhase"
            "DETERMINISTIC_BUILD"
            "TEMPDIR"
            "TMPDIR"
            "PWD"
            "OLDPWD"
            "PATH"
        )

        string(JSON VAR_COUNT ERROR_VARIABLE JSON_ERROR LENGTH "${NIX_DEVELOP_OUTPUT}" "variables")
        foreach(VAR_INDEX RANGE 0 ${VAR_COUNT})
            string(JSON VAR_NAME ERROR_VARIABLE JSON_ERROR MEMBER "${NIX_DEVELOP_OUTPUT}" "variables" ${VAR_INDEX})

            if ("${VAR_NAME}" IN_LIST EXCLUDE_VARS)
                continue()
            endif()

            string(JSON VAR_TYPE ERROR_VARIABLE JSON_ERROR GET "${NIX_DEVELOP_OUTPUT}" "variables" "${VAR_NAME}" "type")

            if(NOT "${VAR_TYPE}" STREQUAL "exported")
                continue()
            endif()

            LoadNixEnvVar("${VAR_NAME}")
        endforeach()
    endfunction()

    function(LoadNixProgToCMakeCache LOAD_VAR CACHE_VAR)
        string(JSON VAR_VALUE ERROR_VARIABLE JSON_ERROR GET "${NIX_DEVELOP_OUTPUT}" "variables" "${LOAD_VAR}" "value")
        if("${JSON_ERROR}" STREQUAL "NOTFOUND")
            message(STATUS "Setting ${CACHE_VAR} to ${VAR_VALUE}")
            find_program(tmp NAMES "${VAR_VALUE}" NO_CACHE)
            set(${CACHE_VAR} "${tmp}" CACHE FILEPATH "Path to ${LOAD_VAR}" FORCE)
            message(STATUS "Found ${CACHE_VAR} at ${tmp}")
        endif()
    endfunction()

    LoadNixPathVar()

    #LoadAllEnvVars()

    LoadNixEnvVar("PKG_CONFIG_PATH")
    LoadNixEnvVar("PYTHONPATH")
    LoadNixEnvVar("CMAKE_INCLUDE_PATH")
    LoadNixEnvVar("CMAKE_LIBRARY_PATH")
    LoadNixEnvVar("CMAKE_PREFIX_PATH")
    LoadNixEnvVar("QMAKE")
    LoadNixEnvVar("QMAKEPATH")
    LoadNixEnvVar("QTTOOLSPATH")
    LoadNixEnvVar("QT_ADDITIONAL_PACKAGES_PREFIX_PATH")

    # TODO: Make this work better - there's some weirdness when specifying the compiler from the toolchain.
    # Really QtCreator shouldn't be specifying an empty compiler.
    LoadNixProgToCMakeCache("CXX" CMAKE_CXX_COMPILER)
    LoadNixProgToCMakeCache("CC"  CMAKE_C_COMPILER)

    message(STATUS "Environment loaded.")

    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR x86_64)

    set(ENV{NIX_PROFILE_LOADED} "1")
endif()
