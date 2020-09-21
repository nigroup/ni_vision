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
 * $Id: pcl_histogram_visualizer.h 31709 2010-08-11 08:11:54Z rusu $
 *
 */
#ifndef PCL_PCL_HISTOGRAM_VISUALIZER_H_
#define PCL_PCL_HISTOGRAM_VISUALIZER_H_

#include <pcl_visualization/interactor.h>
#include <pcl_visualization/interactor_style.h>
#include <pcl_visualization/common/common.h>
#include <pcl_visualization/common/ren_win_interact_map.h>
#include <vtkCommand.h>
#include <vtkXYPlotActor.h>
#include <vtkTextProperty.h>
#include <vtkProperty2D.h>
#include <vtkFieldData.h>
#include <vtkDoubleArray.h>

namespace pcl_visualization
{
  //////////////////////////////////////////////////////////////////////////////////////////////
  /** \brief PCL histogram visualizer main class. 
    * \author Radu Bogdan Rusu
    */
  class PCLHistogramVisualizer
  {
    public:
      /** \brief PCL histogram visualizer constructor. */
      PCLHistogramVisualizer ();

      /** \brief Spin once method. Calls the interactor and updates the screen once. 
       *  \param time - How long (in ms) should the visualization loop be allowed to run.
       *  \param force_redraw - if false it might return without doing anything if the interactor's
       *                        framerate does not require a redraw yet.
       */
      void spinOnce (int time = 1, bool force_redraw = false);
      
      /** \brief Spin method. Calls the interactor and runs an internal loop. */
      void spin ();
      
      /** \brief Set the viewport's background color.
        * \param r the red component of the RGB color
        * \param g the green component of the RGB color
        * \param b the blue component of the RGB color
        * \param viewport the view port (default: all)
        */
      void setBackgroundColor (const double &r, const double &g, const double &b, int viewport = 0);

      /** \brief Add a histogram feature to screen as a separate window.
        * \param cloud the PointCloud dataset containing the histogram
        * \param hsize the length of the histogram
        * \param id the point cloud object id (default: cloud)
        * \param win_width the size of the window (width)
        * \param win_height the size of the window (width)
        */
      template <typename PointT> bool addFeatureHistogram (const pcl::PointCloud<PointT> &cloud, int hsize, const std::string &id = "cloud", int win_width = 640, int win_height = 200);

      /** \brief Add a histogram feature to screen as a separate window.
        * \param cloud the PointCloud dataset containing the histogram
        * \param field_name the field name containing the histogram
        * \param id the point cloud object id (default: cloud)
        * \param win_width the size of the window (width)
        * \param win_height the size of the window (width)
        */
      bool addFeatureHistogram (const sensor_msgs::PointCloud2 &cloud, const std::string &field_name, const std::string &id = "cloud", int win_width = 640, int win_height = 200);

      /** \brief Set the Y range to minp-maxp for all histograms.
        * \param minp the minimum Y range
        * \param maxp the maximum Y range
        */
      void setGlobalYRange (float minp, float maxp);

      /** \brief Update all window positions on screen so that they fit. */
      void updateWindowPositions ();

      /** \brief Returns true when the user tried to close the window */
      bool wasStopped ();
      /** \brief Set the stopped flag back to false */
      void resetStoppedFlag ();

    private:
      /** \brief A map of all windows on screen (with their renderers and interactors). */
      RenWinInteractMap wins_;

      struct ExitMainLoopTimerCallback : public vtkCommand
      {
        static ExitMainLoopTimerCallback* New()
        {
          return new ExitMainLoopTimerCallback;
        }
        virtual void Execute(vtkObject* vtkNotUsed(caller), unsigned long event_id, void* call_data)
        {
          if (event_id != vtkCommand::TimerEvent)
            return;
          int timer_id = *(int*)call_data;

          if (timer_id != right_timer_id)
            return;

          // Stop vtk loop and send notification to app to wake it up
          interact->stopLoop ();
        }
        int right_timer_id;
        PCLVisualizerInteractor *interact;
      };
      struct ExitCallback : public vtkCommand
      {
        static ExitCallback* New ()
        {
          return new ExitCallback;
        }
        virtual void Execute (vtkObject* caller, unsigned long event_id, void* call_data)
        {
          if (event_id != vtkCommand::ExitEvent)
            return;

          interact->stopped = true;
        }
        PCLVisualizerInteractor *interact;
      };

      /** \brief Callback object enabling us to leave the main loop, when a timer fires. */
      vtkSmartPointer<ExitMainLoopTimerCallback> exit_main_loop_timer_callback_;
      vtkSmartPointer<ExitCallback> exit_callback_;

  };
}

#include "libpcl_visualization/histogram_visualizer.hpp"

#endif
