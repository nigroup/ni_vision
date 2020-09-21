/* Auto-generated by genmsg_cpp for file /home/ni/ros_overlays/zzz_packages/pcl/msg/Vertices.msg */
#ifndef PCL_MESSAGE_VERTICES_H
#define PCL_MESSAGE_VERTICES_H
#include <string>
#include <vector>
#include <map>
#include <ostream>
#include "ros/serialization.h"
#include "ros/builtin_message_traits.h"
#include "ros/message_operations.h"
#include "ros/time.h"

#include "ros/macros.h"

#include "ros/assert.h"


namespace pcl
{
template <class ContainerAllocator>
struct Vertices_ {
  typedef Vertices_<ContainerAllocator> Type;

  Vertices_()
  : vertices()
  {
  }

  Vertices_(const ContainerAllocator& _alloc)
  : vertices(_alloc)
  {
  }

  typedef std::vector<uint32_t, typename ContainerAllocator::template rebind<uint32_t>::other >  _vertices_type;
  std::vector<uint32_t, typename ContainerAllocator::template rebind<uint32_t>::other >  vertices;


  ROS_DEPRECATED uint32_t get_vertices_size() const { return (uint32_t)vertices.size(); }
  ROS_DEPRECATED void set_vertices_size(uint32_t size) { vertices.resize((size_t)size); }
  ROS_DEPRECATED void get_vertices_vec(std::vector<uint32_t, typename ContainerAllocator::template rebind<uint32_t>::other > & vec) const { vec = this->vertices; }
  ROS_DEPRECATED void set_vertices_vec(const std::vector<uint32_t, typename ContainerAllocator::template rebind<uint32_t>::other > & vec) { this->vertices = vec; }
private:
  static const char* __s_getDataType_() { return "pcl/Vertices"; }
public:
  ROS_DEPRECATED static const std::string __s_getDataType() { return __s_getDataType_(); }

  ROS_DEPRECATED const std::string __getDataType() const { return __s_getDataType_(); }

private:
  static const char* __s_getMD5Sum_() { return "39bd7b1c23763ddd1b882b97cb7cfe11"; }
public:
  ROS_DEPRECATED static const std::string __s_getMD5Sum() { return __s_getMD5Sum_(); }

  ROS_DEPRECATED const std::string __getMD5Sum() const { return __s_getMD5Sum_(); }

private:
  static const char* __s_getMessageDefinition_() { return "# List of point indices\n\
uint32[] vertices\n\
\n\
"; }
public:
  ROS_DEPRECATED static const std::string __s_getMessageDefinition() { return __s_getMessageDefinition_(); }

  ROS_DEPRECATED const std::string __getMessageDefinition() const { return __s_getMessageDefinition_(); }

  ROS_DEPRECATED virtual uint8_t *serialize(uint8_t *write_ptr, uint32_t seq) const
  {
    ros::serialization::OStream stream(write_ptr, 1000000000);
    ros::serialization::serialize(stream, vertices);
    return stream.getData();
  }

  ROS_DEPRECATED virtual uint8_t *deserialize(uint8_t *read_ptr)
  {
    ros::serialization::IStream stream(read_ptr, 1000000000);
    ros::serialization::deserialize(stream, vertices);
    return stream.getData();
  }

  ROS_DEPRECATED virtual uint32_t serializationLength() const
  {
    uint32_t size = 0;
    size += ros::serialization::serializationLength(vertices);
    return size;
  }

  typedef boost::shared_ptr< ::pcl::Vertices_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pcl::Vertices_<ContainerAllocator>  const> ConstPtr;
  boost::shared_ptr<std::map<std::string, std::string> > __connection_header;
}; // struct Vertices
typedef  ::pcl::Vertices_<std::allocator<void> > Vertices;

typedef boost::shared_ptr< ::pcl::Vertices> VerticesPtr;
typedef boost::shared_ptr< ::pcl::Vertices const> VerticesConstPtr;


template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const  ::pcl::Vertices_<ContainerAllocator> & v)
{
  ros::message_operations::Printer< ::pcl::Vertices_<ContainerAllocator> >::stream(s, "", v);
  return s;}

} // namespace pcl

namespace ros
{
namespace message_traits
{
template<class ContainerAllocator> struct IsMessage< ::pcl::Vertices_<ContainerAllocator> > : public TrueType {};
template<class ContainerAllocator> struct IsMessage< ::pcl::Vertices_<ContainerAllocator>  const> : public TrueType {};
template<class ContainerAllocator>
struct MD5Sum< ::pcl::Vertices_<ContainerAllocator> > {
  static const char* value() 
  {
    return "39bd7b1c23763ddd1b882b97cb7cfe11";
  }

  static const char* value(const  ::pcl::Vertices_<ContainerAllocator> &) { return value(); } 
  static const uint64_t static_value1 = 0x39bd7b1c23763dddULL;
  static const uint64_t static_value2 = 0x1b882b97cb7cfe11ULL;
};

template<class ContainerAllocator>
struct DataType< ::pcl::Vertices_<ContainerAllocator> > {
  static const char* value() 
  {
    return "pcl/Vertices";
  }

  static const char* value(const  ::pcl::Vertices_<ContainerAllocator> &) { return value(); } 
};

template<class ContainerAllocator>
struct Definition< ::pcl::Vertices_<ContainerAllocator> > {
  static const char* value() 
  {
    return "# List of point indices\n\
uint32[] vertices\n\
\n\
";
  }

  static const char* value(const  ::pcl::Vertices_<ContainerAllocator> &) { return value(); } 
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

template<class ContainerAllocator> struct Serializer< ::pcl::Vertices_<ContainerAllocator> >
{
  template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
  {
    stream.next(m.vertices);
  }

  ROS_DECLARE_ALLINONE_SERIALIZER;
}; // struct Vertices_
} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pcl::Vertices_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const  ::pcl::Vertices_<ContainerAllocator> & v) 
  {
    s << indent << "vertices[]" << std::endl;
    for (size_t i = 0; i < v.vertices.size(); ++i)
    {
      s << indent << "  vertices[" << i << "]: ";
      Printer<uint32_t>::stream(s, indent + "  ", v.vertices[i]);
    }
  }
};


} // namespace message_operations
} // namespace ros

#endif // PCL_MESSAGE_VERTICES_H

