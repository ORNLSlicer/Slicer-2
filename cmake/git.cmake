include_guard(DIRECTORY)

function(FetchRevHash)
    find_package(Git)

    if(Git_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            OUTPUT_VARIABLE REV_HASH
            ERROR_VARIABLE REV_FETCH_HASH_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(NOT REV_FETCH_HASH_ERROR STREQUAL "")
            message(WARNING "Failed to fetch revision hash - git reports the following:\n${REV_FETCH_HASH_ERROR}")
            set(REV_HASH "unavailable")
        else()
            execute_process(
                COMMAND ${GIT_EXECUTABLE} diff --quiet
                WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                RESULT_VARIABLE REV_DIRTY
            )

            if(REV_DIRTY)
                string(APPEND REV_HASH "-dirty")
            endif()
        endif()
    else()
        message(WARNING "Unable to find git - cannot fetch revision hash.")
        set(REV_HASH "<unavailable>")
    endif()

    message(STATUS "Configure-time revision hash: <${REV_HASH}>")

    set(RESULT_VAR "${REV_HASH}" PARENT_SCOPE)
endfunction()
