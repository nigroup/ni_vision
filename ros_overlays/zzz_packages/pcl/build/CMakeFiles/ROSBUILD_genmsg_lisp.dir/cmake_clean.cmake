FILE(REMOVE_RECURSE
  "../msg_gen"
  "../src/pcl/msg"
  "../msg_gen"
  "CMakeFiles/ROSBUILD_genmsg_lisp"
  "../msg_gen/lisp/Vertices.lisp"
  "../msg_gen/lisp/_package.lisp"
  "../msg_gen/lisp/_package_Vertices.lisp"
  "../msg_gen/lisp/ModelCoefficients.lisp"
  "../msg_gen/lisp/_package.lisp"
  "../msg_gen/lisp/_package_ModelCoefficients.lisp"
  "../msg_gen/lisp/PointIndices.lisp"
  "../msg_gen/lisp/_package.lisp"
  "../msg_gen/lisp/_package_PointIndices.lisp"
  "../msg_gen/lisp/PolygonMesh.lisp"
  "../msg_gen/lisp/_package.lisp"
  "../msg_gen/lisp/_package_PolygonMesh.lisp"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/ROSBUILD_genmsg_lisp.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
