FILE(REMOVE_RECURSE
  "../msg_gen"
  "../src/pcl/msg"
  "../msg_gen"
  "CMakeFiles/ROSBUILD_genmsg_py"
  "../src/pcl/msg/__init__.py"
  "../src/pcl/msg/_Vertices.py"
  "../src/pcl/msg/_ModelCoefficients.py"
  "../src/pcl/msg/_PointIndices.py"
  "../src/pcl/msg/_PolygonMesh.py"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/ROSBUILD_genmsg_py.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
