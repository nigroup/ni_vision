# ----------------------------------------------------------------------------
# ELM Libraries (must preceed OpenCV)
# ----------------------------------------------------------------------------
# ELM Layers describe the model pipeline
# ----------------------------------------------------------------------------
# find_package ELM
# Defines: ELM_FOUND, ELM_INCLUDE_DIRS, ELM_LIBS
#
# the components we're currently interested in: all
#
# ----------------------------------------------------------------------------
if(DEFINED ELM_DIR)

    status("ELM_DIR=${ELM_DIR}")
    find_package(ELM REQUIRED PATHS ${ELM_DIR})
    if(ELM_FOUND)
    
        list(APPEND ${ROOT_PROJECT}_INCLUDE_DIRS ${ELM_INCLUDE_DIRS})
        list(APPEND ${ROOT_PROJECT}_LIBS ${ELM_LIBS})
        
    else(ELM_FOUND)
        
        message(SEND_ERROR "Failed to find ELM. Double check that \"ELM_DIR\" to the root build directory of SEM libraries.")
    
    endif(ELM_FOUND)
    
else(DEFINED ELM_DIR)
    set(ELM_DIR "" CACHE PATH "Root directory for ELM build directory." )
    message(FATAL_ERROR "\"ELM_DIR\" not set. Please provide the path to the root build directory of the ELM libraries.")
endif(DEFINED ELM_DIR)

