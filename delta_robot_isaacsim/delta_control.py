#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
優化後的Delta機器人控制器
使用delta_motion中的軌跡規劃功能
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Bool
from delta_robot_isaacsim.delta_motion import DeltaMotion, Point as DeltaPoint
import math
from rclpy.callback_groups import ReentrantCallbackGroup
import time

class DeltaRobotController(Node):
    def __init__(self):
        super().__init__('delta_robot_controller')
        
        # 初始化運動控制器（包含軌跡規劃功能）
        self.motion_controller = DeltaMotion()
        
        # 創建回調組
        self.callback_group = ReentrantCallbackGroup()
        
        # 創建訂閱者和發布者
        self.position_sub = self.create_subscription(
            Float32MultiArray,
            '/delta_x/target_array',
            self.trajectory_callback,
            10,
            callback_group=self.callback_group
        )
        
        self.angle_pub = self.create_publisher(
            JointState,
            '/target',
            10
        )
        
        # 新增執行完成狀態發布者
        self.execution_complete_pub = self.create_publisher(
            Bool,
            'delta_execution_complete',
            10
        )
        
        # 初始化 joint state 消息
        self.joint_state = JointState()
        self.joint_state.name = ['part_reducer1________377', 'part_reducer2________334', 'part_reducer3________362']
        self.joint_state.position = [0.0, 0.0, 0.0]
        self.joint_state.velocity = [0.0, 0.0, 0.0]
        self.joint_state.effort = [0.0, 0.0, 0.0]
        
        # 控制狀態
        self.is_executing = False
        self.hold_mode = False
        
        # 創建定時器
        self.timer = self.create_timer(
            1.0 / self.motion_controller.control_frequency, 
            self.control_loop
        )
        
        self.get_logger().info("優化後的Delta機器人控制器已初始化完成")
    
    def publish_angles(self, angles):
        """發布馬達角度"""
        # 角度已經是弧度，直接使用
        angle_values = [
            round(angles.Theta1, 6),
            round(angles.Theta2, 6),
            round(angles.Theta3, 6)
        ]
        
        if not self.motion_controller.check_angle_limits(angles):
            self.get_logger().error(f"角度超出範圍: θ1={math.degrees(angles.Theta1):.2f}°, θ2={math.degrees(angles.Theta2):.2f}°, θ3={math.degrees(angles.Theta3):.2f}°")
            return False
        
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.joint_state.position = angle_values
        self.angle_pub.publish(self.joint_state)
        return True
    
    def trajectory_callback(self, msg):
        """處理接收到的軌跡點序列"""
        if len(msg.data) % 3 != 0:
            self.get_logger().error("數據格式錯誤：點的數量必須是3的倍數")
            return
            
        points = []
        for i in range(0, len(msg.data), 3):
            point = DeltaPoint(
                msg.data[i],      # X
                msg.data[i+1],    # Y
                msg.data[i+2]     # Z
            )
            points.append(point)
            
        self.get_logger().info(f"收到軌跡點序列，共 {len(points)} 個點")
        self.execute_trajectory(points)

    def execute_trajectory(self, points):
        """執行軌跡點序列（批次處理模式）"""
        if self.is_executing:
            self.get_logger().warn("正在執行其他軌跡，請稍後再試")
            return False
        
        self.is_executing = True
        self.hold_mode = False
        
        try:
            # 1. 首先移動到待命位置
            self.get_logger().info("開始執行軌跡，首先移動到待命位置...")
            if not self.move_to_standby_position():
                self.get_logger().error("移動到待命位置失敗，中止執行")
                self.is_executing = False
                return False
            
            # 2. 檢查點數是否為3的倍數（批次處理）
            if len(points) % 3 != 0:
                self.get_logger().error(f"點數 {len(points)} 不是3的倍數，無法進行批次處理")
                self.is_executing = False
                return False
            
            # 3. 按3點一組進行批次處理
            num_groups = len(points) // 3
            self.get_logger().info(f"開始批次處理，共 {num_groups} 組，每組3個點")
            
            for group_idx in range(num_groups):
                self.get_logger().info(f"執行第 {group_idx + 1}/{num_groups} 組")
                
                # 獲取當前組的3個點
                group_points = points[group_idx * 3:(group_idx + 1) * 3]
                
                # 執行3點序列：上方(-550) → 目標(-625) → 拔起(-550)
                for point_idx, point in enumerate(group_points):
                    point_start_time = time.time()
                    self.get_logger().info(f"執行第 {group_idx + 1} 組第 {point_idx + 1} 點: X={point.X:.2f}, Y={point.Y:.2f}, Z={point.Z:.2f}")
                    
                    # 計算逆運動學
                    angles = self.motion_controller.kinematics.inverse_kinematics(point)
                    
                    if angles is None:
                        self.get_logger().error(f"第 {group_idx + 1} 組第 {point_idx + 1} 點逆運動學計算失敗，點不可達")
                        self.is_executing = False
                        return False
                    
                    # 檢查角度範圍
                    if not self.motion_controller.check_angle_limits(angles):
                        self.get_logger().error(f"第 {group_idx + 1} 組第 {point_idx + 1} 點角度超出限制")
                        self.is_executing = False
                        return False
                    
                    # 執行平滑軌跡
                    success = self.execute_smooth_motion(angles)
                    if not success:
                        self.get_logger().error(f"第 {group_idx + 1} 組第 {point_idx + 1} 點軌跡執行失敗")
                        self.is_executing = False
                        return False
                    
                    point_execution_time = time.time() - point_start_time
                    self.get_logger().info(f"第 {group_idx + 1} 組第 {point_idx + 1} 點執行完成，耗時: {point_execution_time:.2f} 秒")
                
                # 每組完成後移動到丟棄位置
                self.get_logger().info(f"第 {group_idx + 1} 組完成，移動到丟棄位置...")
                if not self.move_to_drop_position():
                    self.get_logger().error(f"第 {group_idx + 1} 組完成後移動到丟棄位置失敗")
                    self.is_executing = False
                    return False
                
                self.get_logger().info(f"第 {group_idx + 1} 組批次處理完成")
            
            # 4. 所有批次完成後返回待命位置
            self.get_logger().info("所有批次完成，返回待命位置...")
            if not self.move_to_standby_position():
                self.get_logger().error("返回待命位置失敗")
                self.is_executing = False
                return False
            
            # 5. 發布執行完成狀態
            self.publish_execution_complete()
            
            self.get_logger().info("所有軌跡點執行完成")
            self.hold_mode = True
            self.is_executing = False
            return True
            
        except Exception as e:
            self.get_logger().error(f"執行軌跡時發生錯誤: {str(e)}")
            self.is_executing = False
            return False
            
        finally:
            self.is_executing = False
    
    def execute_smooth_motion(self, target_angles):
        """執行平滑運動到目標角度"""
        try:
            # 將 Angle 對象轉換為弧度列表
            target_angles_list = [
                target_angles.Theta1,  # 已經是弧度
                target_angles.Theta2,  # 已經是弧度
                target_angles.Theta3   # 已經是弧度
            ]
            
            # 使用delta_motion中的軌跡規劃
            trajectories = self.motion_controller.execute_smooth_motion(
                target_angles_list,  # 轉換後的列表
                self.joint_state.position  # 當前關節位置（弧度）
            )
            
            if trajectories is None:
                self.get_logger().error("軌跡規劃失敗")
                return False
            
            # 執行軌跡並等待完成
            success = self._execute_trajectories(trajectories, target_angles_list)
            
            return success
            
        except Exception as e:
            self.get_logger().error(f"執行平滑運動時發生錯誤: {str(e)}")
            return False
    
    def _execute_trajectories(self, trajectories, target_angles):
        """執行多關節軌跡"""
        while not self.motion_controller.get_trajectory_status(trajectories):
            current_time = time.time()
            
            # 使用delta_motion中的進度更新
            current_positions = self.motion_controller.update_trajectory_progress(
                trajectories, current_time, current_time  # 傳入相同的時間值
            )
            
            # current_positions 已經是弧度，直接使用
            self.joint_state.position = current_positions
            
            # 更新並發布關節狀態
            self.joint_state.header.stamp = self.get_clock().now().to_msg()
            self.angle_pub.publish(self.joint_state)
            
            # 控制頻率
            time.sleep(1.0 / self.motion_controller.control_frequency)
        
        # 發送精確的最終位置
        self._send_final_position(target_angles)
        self.is_executing = False

        return True
    
    def _send_final_position(self, target_angles):
        """發送最終精確位置"""
        # target_angles 已經是弧度，直接使用
        final_angles = target_angles.copy()
        
        self.joint_state.position = final_angles
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.angle_pub.publish(self.joint_state)
    
    def move_to_standby_position(self):
        """移動到待命位置"""
        standby_point = self.motion_controller.position_manager.get_standby_position()
        self.get_logger().info(f"移動到待命位置: ({standby_point.X}, {standby_point.Y}, {standby_point.Z})")
        
        # 計算逆運動學
        angles = self.motion_controller.kinematics.inverse_kinematics(standby_point)
        if angles is None:
            self.get_logger().error("待命位置逆運動學計算失敗")
            return False
        
        # 檢查角度範圍
        if not self.motion_controller.check_angle_limits(angles):
            self.get_logger().error("待命位置角度超出限制")
            return False
        
        # 執行平滑運動
        success = self.execute_smooth_motion(angles)
        if success:
            self.get_logger().info("成功移動到待命位置")
        else:
            self.get_logger().error("移動到待命位置失敗")
        
        return success
    
    def move_to_drop_position(self):
        """移動到丟棄位置"""
        drop_point = self.motion_controller.position_manager.get_drop_position()
        self.get_logger().info(f"移動到丟棄位置: ({drop_point.X}, {drop_point.Y}, {drop_point.Z})")
        
        # 計算逆運動學
        angles = self.motion_controller.kinematics.inverse_kinematics(drop_point)
        if angles is None:
            self.get_logger().error("丟棄位置逆運動學計算失敗")
            return False
        
        # 檢查角度範圍
        if not self.motion_controller.check_angle_limits(angles):
            self.get_logger().error("丟棄位置角度超出限制")
            return False
        
        # 執行平滑運動
        success = self.execute_smooth_motion(angles)
        if success:
            self.get_logger().info("成功移動到丟棄位置")
        else:
            self.get_logger().error("移動到丟棄位置失敗")
        
        return success
    
    def publish_execution_complete(self):
        """發布執行完成狀態"""
        complete_msg = Bool()
        complete_msg.data = True
        self.execution_complete_pub.publish(complete_msg)
        self.get_logger().info("發布執行完成狀態")

    def control_loop(self):
        """主控制迴路"""
        if self.hold_mode and not self.is_executing:
            self.joint_state.header.stamp = self.get_clock().now().to_msg()
            self.angle_pub.publish(self.joint_state)

def main(args=None):
    rclpy.init(args=args)
    node = DeltaRobotController()  # 或 TrajectoryPlanNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("收到中斷信號，正在關閉...")
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            # 忽略關閉時的錯誤
            pass

if __name__ == '__main__':
    main()