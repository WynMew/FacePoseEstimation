project(WynEye)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenCV REQUIRED)
MESSAGE("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("OPENCV_LIBRARIES: ${OpenCV_LIBS}")
find_package(dlib REQUIRED)
MESSAGE("DLIB_INCLUDE_DIRS: ${DLIB_INCLUDE_DIR}")
MESSAGE("DLIB_LIBRARIES: ${DLIB_LIB_DIR}")
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Threads REQUIRED)


INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS}
 	/usr/local/include/
)
#vision executable
add_executable (FacePose webcam_face_pose.cpp)
target_link_libraries(FacePose ${OpenCV_LIBS} dlib::dlib Threads::Threads)
