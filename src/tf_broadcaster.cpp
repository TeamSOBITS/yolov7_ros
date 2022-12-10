#include <iostream>
#include <cstdlib>
#include <fstream>
#include <time.h>
#include <sstream>
#include <stdlib.h>
#include <unistd.h>
#include <limits>
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/console.h>

//for image input
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Bool.h>
#include <ros/package.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>

// PCL specific includes
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>

#include <darknet_dnn/BoundingBox.h>
#include <darknet_dnn/BoundingBoxes.h>

#include <float.h>
#define ARRAY_LENGTH(array) (sizeof(array) / sizeof(array[0]))

class tf_broadcaster
{
  ros::NodeHandle nh;
  ros::Subscriber sub_ctrl;
  ros::Subscriber sub_rect;
  ros::Subscriber sub_cloud;

  tf::TransformBroadcaster br;
  tf::TransformListener listener;

  std::string cloud_topic_name;
  std::string camera_frame_name;
  bool execute_flag;
  bool get_cloud_flag;
  pcl::PointCloud<pcl::PointXYZ> cloud_local;


public:
  tf_broadcaster(){

    this->get_cloud_flag = false;
    ros::param::get("execute_default", execute_flag);
    ros::param::get("cloud_topic_name", cloud_topic_name);
    ros::param::get("camera_frame_name", camera_frame_name);
    sub_ctrl = nh.subscribe ("detect_ctrl", 1, &tf_broadcaster::ctrl_cb, this);
    sub_rect = nh.subscribe ("objects_rect", 1, &tf_broadcaster::rect_cb, this);
    sub_cloud = nh.subscribe (cloud_topic_name, 1, &tf_broadcaster::cloud_cb, this);

    ROS_INFO("tf_broadcaster initialize ok");
  }//tf_broadcaster

  ~tf_broadcaster(){}


  void ctrl_cb(const std_msgs::Bool& input){
    execute_flag = input.data;
    if(execute_flag == true){
      ROS_INFO("darknet_dnn_tf_broadcaster -> Start");
    }
    else{
      ROS_INFO("darknet_dnn_tf_broadcaster -> Stopped");
    }
  }//contrl_CB



  void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input){
    if(execute_flag == false){	return;	}

    pcl::fromROSMsg(*input, cloud_local);

    if(cloud_local.points.size() == 0){
      ROS_ERROR("NO point cloud");
      this->get_cloud_flag = false;
      return;
    }
    this->get_cloud_flag = true;
  }

  void rect_cb(const darknet_dnn::BoundingBoxes& msg){
    if(this->get_cloud_flag == false){
      ROS_ERROR("tf_broadcaster -> waiting cloud");
      return;
    }
    //base_footprint基準のpointcloudに変換
    pcl::PointCloud<pcl::PointXYZ> cloud_transform;


    //まず最初に座標変換可能か確認
    std::string base_frame_name = "base_footprint";
    bool key = listener.canTransform (base_frame_name, camera_frame_name, ros::Time(0));
    if(!key)
    {
      ROS_WARN_STREAM("tf_broadcaster : pcl canTransform failue\t" << camera_frame_name << " -> " << base_frame_name );
      cloud_transform = cloud_local;//変換できそうにない場合はしょうがないので元の座標系のままで処理する
      base_frame_name = camera_frame_name;//フレーム名を書き換える
    }//if
    else  { pcl_ros::transformPointCloud(base_frame_name, ros::Time(0), cloud_local, camera_frame_name, cloud_transform, listener); }

    for(int i = 0; i < msg.boundingBoxes.size(); i++)
    {
      pcl::PointXYZ object_pt;
      double shortest_distance = DBL_MAX;
      double point_under_rate = 0.6;//オブジェクト画像の下部分は床やテーブルの平面の場合がある。それらの部分が最短距離の点として求まると困るので、下側は探索しないようにする。画像の高さに対するその係数
      double point_upper_rate = 0.9;
      int shortest_distance_y = 0;

      for(int temp_y = msg.boundingBoxes[i].height * (1.0 - point_upper_rate); temp_y < msg.boundingBoxes[i].height * (1.0 - point_under_rate) ; temp_y++ )
      {
        int object_y = msg.boundingBoxes[i].y + temp_y;
        for(int temp_x = 0; temp_x < msg.boundingBoxes[i].width ; temp_x++ )
        {
          int object_x = msg.boundingBoxes[i].x + temp_x;
          if (cloud_transform.points[cloud_transform.width * object_y + object_x].x < shortest_distance)
          {
            shortest_distance = cloud_transform.points[cloud_transform.width * object_y + object_x].x;
            object_pt = cloud_transform.points[cloud_transform.width * object_y + object_x];
            shortest_distance_y = cloud_transform.width * object_y; // Objectのカメラ基準で一番近いところのBoundingBox画像の行(PCLでは一番左のとこ)
          }
        }
      }
      // 同じ高さの点の平均値を算出
      pcl::PointXYZ object_ave_pt;
      double temp_count = 0;
      for(int temp_x = 0; temp_x < msg.boundingBoxes[i].width; temp_x++)
      {
        double object_x = msg.boundingBoxes[i].x + temp_x;
        if(std::isnan(cloud_transform.points[shortest_distance_y + object_x].x) 
        || std::isnan(cloud_transform.points[shortest_distance_y + object_x].y)
        || std::isnan(cloud_transform.points[shortest_distance_y + object_x].z))  { continue; }
        object_ave_pt.x += cloud_transform.points[shortest_distance_y + object_x].x;
        object_ave_pt.y += cloud_transform.points[shortest_distance_y + object_x].y;
        object_ave_pt.z += cloud_transform.points[shortest_distance_y + object_x].z;
        temp_count += 1;
      }
      if (temp_count == 0)  { continue; }
      object_ave_pt.x /= temp_count;
      object_ave_pt.y /= temp_count;
      object_ave_pt.z /= temp_count;
      cloud_transform.points.push_back(object_ave_pt);

      if(std::isnan(object_ave_pt.x) || std::isnan(object_ave_pt.y) || std::isnan(object_ave_pt.z)) { continue; }
      br.sendTransform(tf::StampedTransform(tf::Transform(tf::Quaternion(0, 0, 0, 1), 
                      tf::Vector3(object_ave_pt.x, object_ave_pt.y, object_ave_pt.z)),
                      ros::Time::now(), 
                      base_frame_name, 
                      msg.boundingBoxes[i].Class));
    }
  }
};



int main(int argc, char** argv)
{
  ros::init(argc, argv, "tf_broadcaster");
  tf_broadcaster tf_class;
  ros::spin();
  return 0;
}
