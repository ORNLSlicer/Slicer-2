cmake_minimum_required(VERSION 3.17)

include(contrib/vcpkg.cmake)

# Define project name and base it on C++
project(ornl_slicer_2 CXX)

option(CMAKE_VERBOSE_MAKEFILE "Verbose MAKE Files" ON)

# Set compile time and pass it as a compiler flag
string(TIMESTAMP APP_COMPILE_TIME "%m-%d-%Y")
add_definitions(-DAPP_COMPILE_TIME="${APP_COMPILE_TIME}")
message(STATUS "App compile date set to: ${APP_COMPILE_TIME}")

# Find  Qt Packages
find_package(Qt5 REQUIRED COMPONENTS
  Core Gui Widgets Concurrent OpenGL Network Charts
)
message(STATUS "Qt Libraries Found")

# Add Qt libs to the running list
list(APPEND libs Qt5::Gui Qt5::Widgets Qt5::OpenGL Qt5::Core Qt5::Concurrent Qt5::Network Qt5::Charts)

# Find all project files.
file(GLOB_RECURSE SOURCES "src/**.cpp")
file(GLOB_RECURSE HEADERS "include/**.h")
file(GLOB_RECURSE RESOURCES "resources/**.qrc")
file(GLOB_RECURSE UI "ui/**.ui")

# Wrap Qt5 objects to generate usable code.
qt5_wrap_cpp(MOC_HEADERS ${HEADERS})
qt5_wrap_ui(UI_HEADERS ${UI})
qt5_add_resources(RCC ${RESOURCES})

# Tell executable where to find headers.
include_directories(PUBLIC "include" ${CMAKE_BINARY_DIR})

find_package(assimp        REQUIRED)
find_package(Boost         REQUIRED COMPONENTS system)
find_package(CGAL          REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(VTK           REQUIRED)
find_package(TBB           REQUIRED)
find_package(OPENMP        REQUIRED)

# Find Eigen
add_definitions(-DEIGEN_NO_CUDA=1) # Eigen does not have CUDA support for MSVC yet
find_package(Eigen3 3.3.0 REQUIRED NO_MODULE)
include(CGAL_Eigen3_support)

list(APPEND libs assimp::assimp Boost::system CGAL::CGAL Eigen3::Eigen nlohmann_json ${VTK_LIBRARIES} OpenMP::OpenMP_CXX)

# Check if CGAL was configured with Eigen
if(TARGET CGAL::Eigen3_support)
    message(STATUS "Eigen found and configred with CGAL")
    list(APPEND libs CGAL::Eigen3_support)
else()
    message(FATAL_ERROR "Eigen NOT Found")
endif()


# Find the other libraries in the contrib folder
add_subdirectory(contrib)
add_subdirectory("contrib/ORNL-TCP-Sockets")
list(APPEND libs_include ${ORNL_TCP_SOCKETS_INCLUDE})
list(APPEND libs ${ORNL_TCP_SOCKETS_LIB})

# Add library include dirs
include_directories(${libs_include})

# Set some Win32 Specific Settings.  For now, set GUI type to suppress console.
#IF(WIN32)
#    SET(GUI_TYPE WIN32)
#ENDIF(WIN32)

# If NVCC is found, set flag, enable lang and find files
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    option(USE_CUDA "Enable CUDA Support" OFF)
    if(USE_CUDA)
        enable_language(CUDA)
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10.0.0)
            message(WARNING "Outdated version(${CMAKE_CUDA_COMPILER_VERSION}) of CUDA found. Please update to enable support")
        else()
            message(STATUS "Found CUDA compiler version ${CMAKE_CUDA_COMPILER_VERSION}")
            set(CUDA_VALID ${CMAKE_CUDA_COMPILER_VERSION})

            # Find and add CUDA files (.cu and .cuh)
            file(GLOB_RECURSE CU "src/**.cu")
            file(GLOB_RECURSE CUH "include/**.cuh")
            list(APPEND cuda_files  ${CU} ${CUH})
            if(NOT DEFINED CMAKE_CUDA_STANDARD)
                set(CMAKE_CUDA_STANDARD 11)
                set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            endif()
        endif()
    endif()
endif(CMAKE_CUDA_COMPILER)

# Build the executable using Qt GUI files, resources, heads, CPP files and CUDA files
add_executable(${PROJECT_NAME} ${GUI_TYPE} ${RCC} ${UI_HEADERS} ${MOC_HEADERS} ${SOURCES} ${HEADERS} ${cuda_files})

# If Single Path lib was found set flag
if(SINGLE_PATH_FOUND)
    message(STATUS "Single Path Library Found")
    target_compile_definitions(${PROJECT_NAME} PUBLIC HAVE_SINGLE_PATH)
else()
    message(WARNING "Single Path Library NOT Found")
endif(SINGLE_PATH_FOUND)

# Set wire feed flag if found
if(WIRE_FEED_FOUND)
    message("Wire Feed Library Found")
    target_compile_definitions(${PROJECT_NAME} PUBLIC HAVE_WIRE_FEED)
else()
    message("Wire Feed Library NOT Found")
endif(WIRE_FEED_FOUND)

# CUDA config
if(CUDA_VALID AND USE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PUBLIC NVCC_FOUND) # Set flag for conditional compilation

    # Generate code for all major versions of CUDA since 5.3 (Maxwell)
    # See this page for more: https://en.wikipedia.org/wiki/CUDA
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 52 53 60 61 70 75 80 86 87)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)

    # NVCC Flags
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --relocatable-device-code=true;
            --expt-extended-lambda;
            --default-stream per-thread;
            --disable-warnings>)

    # Find CUDA libs
    find_package(CUDAToolkit)
    list(APPEND libs CUDA::cublas)
endif()

if(WIN32)
    # Needed for large file obj files sizes. This is required due to heavy template lib use in CGAL and BOOST
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -bigobj>)
    target_link_libraries( ${PROJECT_NAME} ${libs}) # Link to the libs
else()
    # Clang doesn't have/need -bigobj but does need -fpermissive
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fpermissive -Wno-deprecated-declarations>)
    target_link_libraries( ${PROJECT_NAME} ${libs}) # Link to the libs
endif()

# Copy user guide and setting templates to output dir if necessary
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/doc/Slicer_2_User_Guide.pdf" "$<TARGET_FILE_DIR:${PROJECT_NAME}>" )
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/templates"                      "$<TARGET_FILE_DIR:${PROJECT_NAME}>/templates" )
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/layerbartemplates"              "$<TARGET_FILE_DIR:${PROJECT_NAME}>/layerbartemplates" )
find_package (Python COMPONENTS Interpreter)
if(Python_FOUND)
    message(STATUS "Python interpreter found")
    option(GEN_MASTER_CONF "Generate master" ON)
    if(GEN_MASTER_CONF)
        message(STATUS "Generating master.conf file automatically")
        execute_process(COMMAND ${Python_EXECUTABLE} ./scripts/OdsToJson.py ${CMAKE_CURRENT_SOURCE_DIR}/mss_for_json.ods ${CMAKE_CURRENT_SOURCE_DIR}/resources/configs/master.conf
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GEN_OK)
        if (GEN_OK AND NOT STATUS EQUAL 0)
            message(WARNING "Automatic master.conf enabled but command failed. Check the above output and correct the issue to enable this feature.")
        endif()
    endif(GEN_MASTER_CONF)
else (Python_FOUND)
    message("Python NOT found. Please install it to automatically generate master.conf file")
endif(Python_FOUND)

option(BUILD_DOC "Build Doxygen documentation" OFF)
find_package(Doxygen)
if (DOXYGEN_FOUND)
    if(BUILD_DOC)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        message(STATUS "Automatic Doxygen generation enabled")

        add_custom_target( doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
    endif(BUILD_DOC)
else (DOXYGEN_FOUND)
  message(WARNING "If you want Doxygen support, you need to install it first")
endif (DOXYGEN_FOUND)
