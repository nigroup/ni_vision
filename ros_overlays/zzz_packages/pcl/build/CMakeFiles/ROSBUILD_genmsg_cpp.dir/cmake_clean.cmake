FILE(REMOVE_RECURSE
  "../msg_gen"
  "../src/pcl/msg"
  "../msg_gen"
  "CMakeFiles/ROSBUILD_genmsg_cpp"
  "../msg_gen/cpp/include/pcl/Vertices.h"
  "../msg_gen/cpp/include/pcl/ModelCoefficients.h"
  "../msg_gen/cpp/include/pcl/PointIndices.h"
  "../msg_gen/cpp/include/pcl/PolygonMesh.h"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/ROSBUILD_genmsg_cpp.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
