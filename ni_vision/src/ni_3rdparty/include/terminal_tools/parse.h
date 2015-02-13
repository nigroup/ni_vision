/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id: parse.h 30713 2010-07-09 20:00:51Z rusu $
 *
 */
#ifndef TERMINAL_TOOLS_PARSE_H_
#define TERMINAL_TOOLS_PARSE_H_

#include <vector>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <boost/algorithm/string.hpp>

namespace terminal_tools
{
  ////////////////////////////////////////////////////////////////////////////////
  /** \brief Parse for a specific given command line argument. Returns the value 
    * sent as a string.
    * \param argc the number of command line arguments
    * \param argv the command line arguments
    * \param str the string value to search for
    * \param val the resultant value
    */
  int parse_argument (int argc, char** argv, const char* str, std::string &val);

  ////////////////////////////////////////////////////////////////////////////////
  /** \brief Parse for a specific given command line argument. Returns the value 
    * sent as a boolean.
    * \param argc the number of command line arguments
    * \param argv the command line arguments
    * \param str the string value to search for
    * \param val the resultant value
    */
  int parse_argument (int argc, char** argv, const char* str, bool &val);

  ////////////////////////////////////////////////////////////////////////////////
  /** \brief Parse for a specific given command line argument. Returns the value 
    * sent as a double.
    * \param argc the number of command line arguments
    * \param argv the command line arguments
    * \param str the string value to search for
    * \param val the resultant value
    */
  int parse_argument (int argc, char** argv, const char* str, double &val);

  ////////////////////////////////////////////////////////////////////////////////
  /** \brief Parse for a specific given command line argument. Returns the value 
    * sent as an int.
    * \param argc the number of command line arguments
    * \param argv the command line arguments
    * \param str the string value to search for
    * \param val the resultant value
    */
  int parse_argument (int argc, char** argv, const char* str, int &val);

  ////////////////////////////////////////////////////////////////////////////////
  /** \brief Parse for a specific given command line argument. Returns the value 
    * sent as an unsigned int.
    * \param argc the number of command line arguments
    * \param argv the command line arguments
    * \param str the string value to search for
    * \param val the resultant value
    */
  int parse_argument (int argc, char** argv, const char* str, unsigned int &val);

  ////////////////////////////////////////////////////////////////////////////////
  /** \brief Parse for specific given command line arguments (2x values comma 
    * separated). Returns the values sent as doubles.
    * \param argc the number of command line arguments
    * \param argv the command line arguments
    * \param str the command line argument to search for
    * \param f the first output value
    * \param s the second output value
    */
  int parse_2x_arguments (int argc, char** argv, const char* str, double &f, double &s, bool debug = true);

  ////////////////////////////////////////////////////////////////////////////////
  /** \brief Parse for specific given command line arguments (2x values comma 
    * separated). Returns the values sent as ints.
    * \param argc the number of command line arguments
    * \param argv the command line arguments
    * \param str the command line argument to search for
    * \param f the first output value
    * \param s the second output value
    */
  int parse_2x_arguments (int argc, char** argv, const char* str, int &f, int &s, bool debug = true);

  ////////////////////////////////////////////////////////////////////////////////
  /** \brief Parse for specific given command line arguments (3x values comma 
    * separated). Returns the values sent as doubles.
    * \param argc the number of command line arguments
    * \param argv the command line arguments
    * \param str the command line argument to search for
    * \param f the first output value
    * \param s the second output value
    * \param t the third output value
    */
  int parse_3x_arguments (int argc, char** argv, const char* str, double &f, double &s, double &t, bool debug = true);

  ////////////////////////////////////////////////////////////////////////////////
  /** \brief Parse for specific given command line arguments (3x values comma 
    * separated). Returns the values sent as ints.
    * \param argc the number of command line arguments
    * \param argv the command line arguments
    * \param str the command line argument to search for
    * \param f the first output value
    * \param s the second output value
    * \param t the third output value
    */
  int parse_3x_arguments (int argc, char** argv, const char* str, int &f, int &s, int &t, bool debug = true);

  ////////////////////////////////////////////////////////////////////////////////
  /** \brief Parse for specific given command line arguments (multiple occurances 
    * of the same command line parameter). Returns the values sent as a vector.
    * \param argc the number of command line arguments
    * \param argv the command line arguments
    * \param str the command line argument to search for
    * \param values the resultant output values
    */
  bool parse_multiple_arguments (int argc, char** argv, const char* str, std::vector<int> &values);

  ////////////////////////////////////////////////////////////////////////////////
  /** \brief Parse for specific given command line arguments (multiple occurances 
    * of the same command line parameter). Returns the values sent as a vector.
    * \param argc the number of command line arguments
    * \param argv the command line arguments
    * \param str the command line argument to search for
    * \param values the resultant output values
    */
  bool parse_multiple_arguments (int argc, char** argv, const char* str, std::vector<double> &values);

  ////////////////////////////////////////////////////////////////////////////////
  /** \brief Parse for specific given command line arguments (multiple occurances 
    * of 2x argument groups, separated by commas). Returns 2 vectors holding the 
    * given values.
    * \param argc the number of command line arguments
    * \param argv the command line arguments
    * \param str the command line argument to search for
    * \param values_f the first vector of output values
    * \param values_s the second vector of output values
    */
  bool parse_multiple_2x_arguments (int argc, char** argv, const char* str, std::vector<double> &values_f, std::vector<double> &values_s);

  ////////////////////////////////////////////////////////////////////////////////
  /** \brief Parse for specific given command line arguments (multiple occurances 
    * of 3x argument groups, separated by commas). Returns 3 vectors holding the 
    * given values.
    * \param argc the number of command line arguments
    * \param argv the command line arguments
    * \param str the command line argument to search for
    * \param values_f the first vector of output values
    * \param values_s the second vector of output values
    * \param values_t the third vector of output values
    */
  bool parse_multiple_3x_arguments (int argc, char** argv, const char* str, std::vector<double> &values_f, std::vector<double> &values_s, std::vector<double> &values_t);

  ////////////////////////////////////////////////////////////////////////////////
  /** \brief Parse command line arguments for file names. Returns a vector with 
    * file names indices.
    * \param argc the number of command line arguments
    * \param argv the command line arguments
    * \param extension to search for
    */
  std::vector<int> parse_file_extension_argument (int argc, char** argv, const std::string &ext);
} 

#endif
