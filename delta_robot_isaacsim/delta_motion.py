import math
import numpy as np
from dataclasses import dataclass
import time
import logging
from typing import Tuple, List


@dataclass
class Point:
    X: float = 0.0
    Y: float = 0.0
    Z: float = 0.0

@dataclass
class Angle:
    Theta1: float = 0.0
    Theta2: float = 0.0
    Theta3: float = 0.0

class PositionManager:
    """位置管理類別，管理待命位置和丟棄位置"""
    
    def __init__(self):
        # 待命位置座標 (300, 300, -550)
        self.standby_position = Point(300.0, 300.0, -550.0)
        # 丟棄位置座標 (與待命位置相同)
        self.drop_position = Point(300.0, 300.0, -550.0)
    
    def get_standby_position(self):
        """獲取待命位置"""
        return self.standby_position
    
    def get_drop_position(self):
        """獲取丟棄位置"""
        return self.drop_position
    
    def is_standby_position(self, point):
        """檢查是否為待命位置"""
        return (abs(point.X - self.standby_position.X) < 0.1 and
                abs(point.Y - self.standby_position.Y) < 0.1 and
                abs(point.Z - self.standby_position.Z) < 0.1)
    
    def is_drop_position(self, point):
        """檢查是否為丟棄位置"""
        return (abs(point.X - self.drop_position.X) < 0.1 and
                abs(point.Y - self.drop_position.Y) < 0.1 and
                abs(point.Z - self.drop_position.Z) < 0.1)

class DeltaKinematics:
    def __init__(self):
        # 三角函數常數
        self.tan30 = 1 / math.sqrt(3)
        self.sin30 = 0.5
        self.cos30 = math.sqrt(3) / 2
        self.tan60 = math.sqrt(3)
        self.sin120 = math.sqrt(3) / 2
        self.cos120 = -0.5
        
        # 機器人參數
        self.RD_RF = 300     # 上臂長度，300mm
        self.RD_RE = 700     # 下臂長度，700mm
        self.RD_F = 334.641  # 固定平台半徑，334.641mm 193.205
        self.RD_E = 207.846  # 移動平台半徑，207.846mm 122.233
        
        # 計算常用值
        self.RD_RF_Pow2 = self.RD_RF * self.RD_RF
        self.RD_RE_Pow2 = self.RD_RE * self.RD_RE
        self._y1_ = -0.5 * self.tan30 * self.RD_F
        self._y0_ = 0.5 * self.tan30 * self.RD_E

    def angle_theta_calculations(self, x0, y0, z0):
        """計算單個馬達的角度"""
        y0 -= self._y0_
        y1 = self._y1_

        # z = a + b*y
        a = (x0*x0 + y0*y0 + z0*z0 + self.RD_RF_Pow2 - self.RD_RE_Pow2 - y1*y1) / (2.0*z0)
        b = (y1 - y0) / z0

        d = -(a + b*y1)*(a + b*y1) + self.RD_RF_Pow2*(b*b + 1.0)

        if d < 0:
            return None

        yj = (y1 - a*b - math.sqrt(d)) / (b*b + 1.0)
        zj = a + b*yj

        theta = math.degrees(math.atan(-zj/(y1 - yj)))
        if yj > y1:
            theta += 180.0

        return theta

    def inverse_kinematics(self, point):
        """逆運動學：從末端位置計算馬達角度"""
        theta1 = self.angle_theta_calculations(point.X, point.Y, point.Z)
        if theta1 is None:
            return None

        theta2 = self.angle_theta_calculations(
            point.X * self.cos120 + point.Y * self.sin120,
            point.Y * self.cos120 - point.X * self.sin120,
            point.Z
        )
        if theta2 is None:
            return None

        theta3 = self.angle_theta_calculations(
            point.X * self.cos120 - point.Y * self.sin120,
            point.Y * self.cos120 + point.X * self.sin120,
            point.Z
        )
        if theta3 is None:
            return None

        # 配合isaacsim將度數轉換為弧度
        angle = Angle(
            math.radians(theta1),
            math.radians(theta2), 
            math.radians(theta3)
        )
        return angle

class TrajectoryPlanner:
    """軌跡規劃器"""
    
    def __init__(self, frequency: float = 200.0):
        self.frequency = frequency
        self.dt = 1.0 / frequency
        
    def wrap_to_pi(self, angle: float) -> float:
        """符合isaacsim將角度包裝到 [-π, π] 範圍內"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def calculate_total_distance(self, theta_goal: float, theta_now: float) -> float:
        """計算總角距"""
        return self.wrap_to_pi(theta_goal - theta_now)
    
    def generate_constant_velocity_trajectory(self, 
                                        theta_start: float, 
                                        theta_end: float, 
                                        velocity: float) -> Tuple[List[float], List[float], float]:
        """生成均速軌跡"""
        total_distance = self.calculate_total_distance(theta_end, theta_start)
        total_time = abs(total_distance) / velocity
        
        num_steps = int(total_time * self.frequency) + 1
        time_sequence = np.linspace(0, total_time, num_steps)
        
        position_sequence = []
        for t in time_sequence:
            if t <= 0:
                pos = theta_start
            elif t >= total_time:
                pos = theta_end
            else:
                pos = theta_start + (total_distance * t / total_time)
            position_sequence.append(pos)
        
        return position_sequence, time_sequence.tolist(), total_time

class DeltaMotion:
    def __init__(self):
        self.kinematics = DeltaKinematics()
        self.trajectory_planner = TrajectoryPlanner(frequency=200.0) 
        self.position_manager = PositionManager()  # 新增位置管理器
        
        # 將初始角度從度數轉換為弧度
        self.current_angle = Angle(
            math.radians(-38.0), 
            math.radians(-38.0), 
            math.radians(-38.0)
        )
        
        # 控制參數
        self.max_velocity = math.pi / 16.0      # π/16 rad/s (11.25°/s)
        self.control_frequency = 200.0          # 200 Hz
    
    def check_angle_limits(self, angle):
        """檢查角度是否在限制範圍內"""
        # 預留0.05度應對極限情況
        min_angle_deg = -38.05
        max_angle_deg = 90.05
        
        theta1_deg = math.degrees(angle.Theta1)
        theta2_deg = math.degrees(angle.Theta2)
        theta3_deg = math.degrees(angle.Theta3)
        
        if (theta1_deg < min_angle_deg or theta1_deg > max_angle_deg or
            theta2_deg < min_angle_deg or theta2_deg > max_angle_deg or
            theta3_deg < min_angle_deg or theta3_deg > max_angle_deg):
            return False
        return True
    
    def execute_smooth_motion(self, target_angles, current_angles):
        """執行平滑運動到目標角度"""
        try:
            # 為每個關節生成均速軌跡
            trajectories = []
            
            for i in range(3):
                start_angle = current_angles[i]  
                end_angle = target_angles[i]     
                
                positions, times, total_time = self.trajectory_planner.generate_constant_velocity_trajectory(
                    start_angle, end_angle, self.max_velocity
                )
                
                # 修正：使用當前時間作為開始時間
                trajectories.append({
                    'positions': positions,
                    'start_time': time.time(),  # 使用當前時間
                    'total_time': total_time,
                    'completed': False,
                    'target_angle': end_angle  # 使用弧度
                })
            
            return trajectories
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
    
    def get_trajectory_status(self, trajectories):
        """獲取軌跡執行狀態"""
        return all(traj['completed'] for traj in trajectories)
    
    def update_trajectory_progress(self, trajectories, current_time, start_time):
        """更新軌跡進度"""
        current_positions = []
        
        for i, traj in enumerate(trajectories):
            if traj['completed']:
                current_positions.append(traj['target_angle'])  
            else:
                # 簡化時間計算：直接使用軌跡的開始時間
                elapsed_time = current_time - traj['start_time']
                
                if elapsed_time >= traj['total_time']:
                    current_positions.append(traj['target_angle'])  
                    traj['completed'] = True
                else:
                    time_ratio = elapsed_time / traj['total_time']
                    num_steps = len(traj['positions'])
                    step_index = int(time_ratio * (num_steps - 1))
                    
                    if step_index >= len(traj['positions']):
                        current_positions.append(traj['target_angle'])  
                    else:
                        current_positions.append(traj['positions'][step_index])  
        
        return current_positions
