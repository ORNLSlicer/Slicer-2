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
            set(TMP "$ENV{PATH}")
            string(APPEND TMP ":${VAR_VALUE}")
            set(ENV{PATH} ${TMP})
        endif()
    endfunction()

    function(LoadNixEnvVar LOAD_VAR)
        string(JSON VAR_VALUE ERROR_VARIABLE JSON_ERROR GET "${NIX_DEVELOP_OUTPUT}" "variables" "${LOAD_VAR}" "value")
        if("${JSON_ERROR}" STREQUAL "NOTFOUND")
            message(STATUS "Setting ${LOAD_VAR}")
            set(ENV{${LOAD_VAR}} ${VAR_VALUE})
        endif()
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

    function(RenameEnvVar OLD_NAME NEW_NAME)
        if(DEFINED ENV{${OLD_NAME}})
            set(ENV{${NEW_NAME}} $ENV{${OLD_NAME}})
            unset(ENV{${OLD_NAME}})
        endif()
    endfunction()

    LoadNixPathVar()

    LoadNixEnvVar("PKG_CONFIG_PATH")
    LoadNixEnvVar("PYTHONPATH")
    LoadNixEnvVar("CMAKE_INCLUDE_PATH")
    LoadNixEnvVar("CMAKE_LIBRARY_PATH")
    LoadNixEnvVar("NIXPKGS_CMAKE_PREFIX_PATH")
    LoadNixEnvVar("QMAKE")
    LoadNixEnvVar("QMAKEPATH")
    LoadNixEnvVar("QTTOOLSPATH")
    LoadNixEnvVar("QT_ADDITIONAL_PACKAGES_PREFIX_PATH")

    # Newer versions of nixpkgs stick their prefix paths into NIXPKGS_CMAKE_PREFIX_PATH, so we need to rename so CMake can find it.
    RenameEnvVar("NIXPKGS_CMAKE_PREFIX_PATH" "CMAKE_PREFIX_PATH")

    # TODO: Make this work better - there's some weirdness when specifying the compiler from the toolchain.
    # Really QtCreator shouldn't be specifying an empty compiler.
    LoadNixProgToCMakeCache("CXX" CMAKE_CXX_COMPILER)
    LoadNixProgToCMakeCache("CC"  CMAKE_C_COMPILER)

    message(STATUS "Environment loaded.")

    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR x86_64)

    set(ENV{NIX_PROFILE_LOADED} "1")
endif()
