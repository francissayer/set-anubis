cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(MartyInstaller)

include(ExternalProject)

set(MARTY_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
set(MARTY_INSTALL_PATH "${MARTY_PATH}/install")

set(MARTY_CORRECT_COMPILER_VERSION "11")  # Remplacez par la version correcte de votre compilateur

# Script pour modifier la version du compilateur dans .setup_script.sh
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/MARTY/modify_compiler_version.sh" 
    "#!/bin/bash\n"
    "sed -i 's/set(CMAKE_CXX_STANDARD 17)/set(CMAKE_CXX_STANDARD 20)/' ${MARTY_PATH}/src/CMakeLists.txt\n"
    # "sed -i 's/defaultCompilerVersion=7/defaultCompilerVersion=${MARTY_CORRECT_COMPILER_VERSION}/' ${MARTY_PATH}/src/MARTY/.setup_script.sh\n"
    # "sed -i 's/\\([^:]\\)\\bsize_t\\b/\\1std::size_t/g' ${MARTY_PATH}/src/MARTY/csl/src/multipermutation.cpp\n"
    # "sed -i 's/\\([^:]\\)\\bsize_t\\b/\\1std::size_t/g' ${MARTY_PATH}/src/MARTY/csl/include/multipermutation.h\n"
    
)
execute_process(COMMAND chmod +x "${CMAKE_CURRENT_BINARY_DIR}/MARTY/modify_compiler_version.sh")

set(CUSTOM_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/MARTY/modify_compiler_version.sh")
set(INSTALL_PATH "${MARTY_PATH}/MARTY_INSTALL")
file(WRITE "${MARTY_PATH}/src/run_setup.sh" 
    "#!/bin/bash\n"
    "cd ${MARTY_PATH}\n"
    "script=\"${MARTY_PATH}/MARTY_INSTALL/lib/libmarty.so\"\n"
    "if [ ! -f \"$script\" ]; then\n"
    "echo \" Installation of marty started \"\n"
    "mkdir MARTY_INSTALL\n"
    "cd src/MARTY\n"
    "mkdir build\n"
    "cd build\n"
    "cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${MARTY_PATH}/MARTY_INSTALL\n"
    "sed -i 's/#include \"abstract.h\"/#include <algorithm>\\n#include \"abstract.h\"/g' ${MARTY_PATH}/src/MARTY/src/csl/abreviation.h\n"
    "sed -i 's/#include \"vector.h\"/#include <algorithm>\\n#include \"vector.h\"/g' ${MARTY_PATH}/src/MARTY/src/csl/diagonalization.h\n"
    "sed -i 's/#pragma once/#include <algorithm>\\n#pragma once/g' ${MARTY_PATH}/src/MARTY/src/csl/hardFactor.h\n"
    "sed -i 's/#include <map>/#include <algorithm>\\n#include <map>/g' ${MARTY_PATH}/src/MARTY/src/csl/index.h\n"
    "sed -i 's/#include \"abstract.h\"/#include <type_traits>\\n#include \"parent.h\"\\n#include \"abstract.h\"/g' ${MARTY_PATH}/src/MARTY/src/csl/replace.h\n"
    "sed -i 's/\\([^:]\\)\\bfind_if\\b/\\1std::find_if/g' ${MARTY_PATH}/src/MARTY/src/csl/variableParent.cpp\n"
    "sed -i 's/#include \"abstract.h\"/#include <algorithm>\\n#include \"abstract.h\"/g' ${MARTY_PATH}/src/MARTY/src/csl/variableParent.h\n"
    "sed -i 's/#include <map>/#include <algorithm>\\n#include <map>/g' ${MARTY_PATH}/src/MARTY/src/grafed/core/latexLink.h\n"
    "sed -i 's/#include <map>/#include <algorithm>\\n#include <map>/g' ${MARTY_PATH}/src/MARTY/src/grafed/gui/latexLink.h\n"
    "make\n"
    "make install\n"
    "else\n"
    "echo \"Installation of Marty already done\" \n"
    "fi\n"
    "bashrc_file=\"$HOME/.bashrc\"\n"
    
    "if ! grep -q \"export PATH=\\\$PATH:${INSTALL_PATH}/bin\" \"\$bashrc_file\"; then
    echo \"export PATH=\\\$PATH:${INSTALL_PATH}/bin\" >> \"\$bashrc_file\"
    fi

    if ! grep -q \"export CPATH=\\\$CPATH:${INSTALL_PATH}/include\" \"\$bashrc_file\"; then
        echo \"export CPATH=\\\$CPATH:${INSTALL_PATH}/include\" >> \"\$bashrc_file\"
    fi

    if ! grep -q \"export C_INCLUDE_PATH=\\\$C_INCLUDE_PATH:${INSTALL_PATH}/include\" \"\$bashrc_file\"; then
        echo \"export C_INCLUDE_PATH=\\\$C_INCLUDE_PATH:${INSTALL_PATH}/include\" >> \"\$bashrc_file\"
    fi

    if ! grep -q \"export LIBRARY_PATH=\\\$LIBRARY_PATH:${INSTALL_PATH}/lib\" \"\$bashrc_file\"; then
        echo \"export LIBRARY_PATH=\\\$LIBRARY_PATH:${INSTALL_PATH}/lib\" >> \"\$bashrc_file\"
    fi

    if ! grep -q \"export LD_LIBRARY_PATH=\\\$LD_LIBRARY_PATH:${INSTALL_PATH}/lib\" \"\$bashrc_file\"; then
        echo \"export LD_LIBRARY_PATH=\\\$LD_LIBRARY_PATH:${INSTALL_PATH}/lib\" >> \"\$bashrc_file\"
    fi

    source \"\$bashrc_file\"

    echo \"Environment variable are uptodate in the .bashrc file.\"
")

execute_process(COMMAND chmod +x "${MARTY_PATH}/src/run_setup.sh")

# set(MARTY_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../ExternalIntegration/MARTY/marty)    
ExternalProject_Add(MARTY
    GIT_REPOSITORY "https://github.com/docbrown1955/marty-public.git"
    GIT_TAG 1.6-beta
    PREFIX ${MARTY_PATH}
    # SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalIntegration/MARTY/src"
    # BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalIntegration/MARTY/build"
    # INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ExternalIntegration/MARTY/install"
    CONFIGURE_COMMAND ""
    COMMAND ${MARTY_PATH}/src/run_setup.sh
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    # COMMAND ${CMAKE_CURRENT_BINARY_DIR}/MARTY/modify_compiler_version.sh
    # CMAKE_ARGS
    #     -DCMAKE_INSTALL_PREFIX="${CMAKE_CURRENT_SOURCE_DIR}/../ExternalIntegration/MARTY/install"
    #     -DCMAKE_BUILD_TYPE=Release
    # BUILD_COMMAND ""
    LOG_DOWNLOAD ON
    LOG_UPDATE ON
    LOG_BUILD ON
)


ExternalProject_Get_Property(MARTY INSTALL_DIR)
message(STATUS "${INSTALL_DIR}")
# # target_include_directories(ExternalIntegrationLib PUBLIC "${INSTALL_DIR}/MARTY_INSTALL/include")
# target_link_libraries(ExternalIntegrationLib PUBLIC "${INSTALL_DIR}/MARTY_INSTALL/lib/libmarty.so")
# # target_link_libraries(ExternalIntegrationLib PUBLIC "${INSTALL_DIR}/MARTY_INSTALL/lib/libcsl.so")
# # target_link_libraries(ExternalIntegrationLib PUBLIC "${INSTALL_DIR}/MARTY_INSTALL/lib/libooptools.so")
# add_dependencies(ExternalIntegrationLib MARTY)

# set_target_properties(ExternalIntegrationLib PROPERTIES PUBLIC_HEADER "2HDMC.h;Interface.h;Softsusy.h")

# target_include_directories(ExternalIntegrationLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


# target_link_libraries(ExternalIntegrationLib PUBLIC CommonLib)
# target_link_libraries(ExternalIntegrationLib PUBLIC ConfiguredModule)