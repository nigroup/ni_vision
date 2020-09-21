/* Auto-generated by genmsg_cpp for file /home/ni/ros_overlays/zzz_packages/pcl/msg/ModelCoefficients.msg */
#ifndef PCL_MESSAGE_MODELCOEFFICIENTS_H
#define PCL_MESSAGE_MODELCOEFFICIENTS_H
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

#include "std_msgs/Header.h"

namespace pcl
{
template <class ContainerAllocator>
struct ModelCoefficients_ {
  typedef ModelCoefficients_<ContainerAllocator> Type;

  ModelCoefficients_()
  : header()
  , values()
  {
  }

  ModelCoefficients_(const ContainerAllocator& _alloc)
  : header(_alloc)
  , values(_alloc)
  {
  }

  typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
   ::std_msgs::Header_<ContainerAllocator>  header;

  typedef std::vector<float, typename ContainerAllocator::template rebind<float>::other >  _values_type;
  std::vector<float, typename ContainerAllocator::template rebind<float>::other >  values;


  ROS_DEPRECATED uint32_t get_values_size() const { return (uint32_t)values.size(); }
  ROS_DEPRECATED void set_values_size(uint32_t size) { values.resize((size_t)size); }
  ROS_DEPRECATED void get_values_vec(std::vector<float, typename ContainerAllocator::template rebind<float>::other > & vec) const { vec = this->values; }
  ROS_DEPRECATED void set_values_vec(const std::vector<float, typename ContainerAllocator::template rebind<float>::other > & vec) { this->values = vec; }
private:
  static const char* __s_getDataType_() { return "pcl/ModelCoefficients"; }
public:
  ROS_DEPRECATED static const std::string __s_getDataType() { return __s_getDataType_(); }

  ROS_DEPRECATED const std::string __getDataType() const { return __s_getDataType_(); }

private:
  static const char* __s_getMD5Sum_() { return "ca27dea75e72cb894cd36f9e5005e93e"; }
public:
  ROS_DEPRECATED static const std::string __s_getMD5Sum() { return __s_getMD5Sum_(); }

  ROS_DEPRECATED const std::string __getMD5Sum() const { return __s_getMD5Sum_(); }

private:
  static const char* __s_getMessageDefinition_() { return "Header header\n\
float32[] values\n\
\n\
\n\
================================================================================\n\
MSG: std_msgs/Header\n\
# Standard metadata for higher-level stamped data types.\n\
# This is generally used to communicate timestamped data \n\
# in a particular coordinate frame.\n\
# \n\
# sequence ID: consecutively increasing ID \n\
uint32 seq\n\
#Two-integer timestamp that is expressed as:\n\
# * stamp.secs: seconds (stamp_secs) since epoch\n\
# * stamp.nsecs: nanoseconds since stamp_secs\n\
# time-handling sugar is provided by the client library\n\
time stamp\n\
#Frame this data is associated with\n\
# 0: no frame\n\
# 1: global frame\n\
string frame_id\n\
\n\
"; }
public:
  ROS_DEPRECATED static const std::string __s_getMessageDefinition() { return __s_getMessageDefinition_(); }

  ROS_DEPRECATED const std::string __getMessageDefinition() const { return __s_getMessageDefinition_(); }

  ROS_DEPRECATED virtual uint8_t *serialize(uint8_t *write_ptr, uint32_t seq) const
  {
    ros::serialization::OStream stream(write_ptr, 1000000000);
    ros::serialization::serialize(stream, header);
    ros::serialization::serialize(stream, values);
    return stream.getData();
  }

  ROS_DEPRECATED virtual uint8_t *deserialize(uint8_t *read_ptr)
  {
    ros::serialization::IStream stream(read_ptr, 1000000000);
    ros::serialization::deserialize(stream, header);
    ros::serialization::deserialize(stream, values);
    return stream.getData();
  }

  ROS_DEPRECATED virtual uint32_t serializationLength() const
  {
    uint32_t size = 0;
    size += ros::serialization::serializationLength(header);
    size += ros::serialization::serializationLength(values);
    return size;
  }

  typedef boost::shared_ptr< ::pcl::ModelCoefficients_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pcl::ModelCoefficients_<ContainerAllocator>  const> ConstPtr;
  boost::shared_ptr<std::map<std::string, std::string> > __connection_header;
}; // struct ModelCoefficients
typedef  ::pcl::ModelCoefficients_<std::allocator<void> > ModelCoefficients;

typedef boost::shared_ptr< ::pcl::ModelCoefficients> ModelCoefficientsPtr;
typedef boost::shared_ptr< ::pcl::ModelCoefficients const> ModelCoefficientsConstPtr;


template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const  ::pcl::ModelCoefficients_<ContainerAllocator> & v)
{
  ros::message_operations::Printer< ::pcl::ModelCoefficients_<ContainerAllocator> >::stream(s, "", v);
  return s;}

} // namespace pcl

namespace ros
{
namespace message_traits
{
template<class ContainerAllocator> struct IsMessage< ::pcl::ModelCoefficients_<ContainerAllocator> > : public TrueType {};
template<class ContainerAllocator> struct IsMessage< ::pcl::ModelCoefficients_<ContainerAllocator>  const> : public TrueType {};
template<class ContainerAllocator>
struct MD5Sum< ::pcl::ModelCoefficients_<ContainerAllocator> > {
  static const char* value() 
  {
    return "ca27dea75e72cb894cd36f9e5005e93e";
  }

  static const char* value(const  ::pcl::ModelCoefficients_<ContainerAllocator> &) { return value(); } 
  static const uint64_t static_value1 = 0xca27dea75e72cb89ULL;
  static const uint64_t static_value2 = 0x4cd36f9e5005e93eULL;
};

template<class ContainerAllocator>
struct DataType< ::pcl::ModelCoefficients_<ContainerAllocator> > {
  static const char* value() 
  {
    return "pcl/ModelCoefficients";
  }

  static const char* value(const  ::pcl::ModelCoefficients_<ContainerAllocator> &) { return value(); } 
};

template<class ContainerAllocator>
struct Definition< ::pcl::ModelCoefficients_<ContainerAllocator> > {
  static const char* value() 
  {
    return "Header header\n\
float32[] values\n\
\n\
\n\
================================================================================\n\
MSG: std_msgs/Header\n\
# Standard metadata for higher-level stamped data types.\n\
# This is generally used to communicate timestamped data \n\
# in a particular coordinate frame.\n\
# \n\
# sequence ID: consecutively increasing ID \n\
uint32 seq\n\
#Two-integer timestamp that is expressed as:\n\
# * stamp.secs: seconds (stamp_secs) since epoch\n\
# * stamp.nsecs: nanoseconds since stamp_secs\n\
# time-handling sugar is provided by the client library\n\
time stamp\n\
#Frame this data is associated with\n\
# 0: no frame\n\
# 1: global frame\n\
string frame_id\n\
\n\
";
  }

  static const char* value(const  ::pcl::ModelCoefficients_<ContainerAllocator> &) { return value(); } 
};

template<class ContainerAllocator> struct HasHeader< ::pcl::ModelCoefficients_<ContainerAllocator> > : public TrueType {};
template<class ContainerAllocator> struct HasHeader< const ::pcl::ModelCoefficients_<ContainerAllocator> > : public TrueType {};
} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

template<class ContainerAllocator> struct Serializer< ::pcl::ModelCoefficients_<ContainerAllocator> >
{
  template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
  {
    stream.next(m.header);
    stream.next(m.values);
  }

  ROS_DECLARE_ALLINONE_SERIALIZER;
}; // struct ModelCoefficients_
} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pcl::ModelCoefficients_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const  ::pcl::ModelCoefficients_<ContainerAllocator> & v) 
  {
    s << indent << "header: ";
s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "values[]" << std::endl;
    for (size_t i = 0; i < v.values.size(); ++i)
    {
      s << indent << "  values[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.values[i]);
    }
  }
};


} // namespace message_operations
} // namespace ros

#endif // PCL_MESSAGE_MODELCOEFFICIENTS_H
