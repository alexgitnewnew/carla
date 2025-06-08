import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import carla
import cv2
from gym import spaces
import time
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random

# Инициализация TensorBoard
writer = SummaryWriter("runs/DDPG_Carla_Obstacle_Avoidance")

# Ornstein-Uhlenbeck шум для исследования
class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# Буфер воспроизведения
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

class CarlaEnvDDPG(gym.Env):
    def __init__(self):
        super(CarlaEnvDDPG, self).__init__()
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.last_location = None
        self.distance = 0

        # Пространство действий: [throttle, steer]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Пространство наблюдений: RGB + семантическая сегментация (64x64x6)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 6), dtype=np.uint8
        )

        self.state = None
        self.vehicle = None
        self.rgb_camera = None
        self.sem_camera = None
        self.lidar = None
        self.collision_sensor = None
        self.collision_flag = False
        self.episode_reward = 0
        self.rgb_image = None
        self.sem_image = None
        self.step_counter = 0
        self.obstacle_side = None
        self.pole_detected = False

        self.fixed_spawn_point = carla.Transform(
            carla.Location(x=80.265495, y=16.907003, z=0.600000),
            carla.Rotation(pitch=0, yaw=0, roll=0)
        )

        # Цвета для семантической сегментации (CityScapesPalette)
        self.semantic_colors = {
            'road': [128, 64, 128],
            'sidewalk': [244, 35, 232],
            'wall': [102, 102, 156],
            'building': [70, 70, 70],
            'car': [0, 0, 142],
            'pedestrian': [0, 0, 220],
            'lane_marking': [157, 234, 50],
            'pole': [153, 153, 153],
            'vegetation': [107, 142, 35],
            'traffic_sign': [220, 220, 0]
        }

    def reset(self):
        self.destroy_actors()
        time.sleep(1)

        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, self.fixed_spawn_point)

        if self.vehicle is None:
            print("[RESET] Failed to spawn vehicle, retrying...")
            return self.reset()

        # RGB камера
        rgb_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute('image_size_x', '640')
        rgb_camera_bp.set_attribute('image_size_y', '480')
        self.rgb_camera = self.world.spawn_actor(
            rgb_camera_bp,
            carla.Transform(carla.Location(x=2.5, z=1.0)),
            attach_to=self.vehicle
        )
        self.rgb_camera.listen(lambda image: self._process_rgb_image(image))

        # Камера семантической сегментации
        sem_camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        sem_camera_bp.set_attribute('image_size_x', '640')
        sem_camera_bp.set_attribute('image_size_y', '480')
        sem_camera_bp.set_attribute('fov', '90')
        self.sem_camera = self.world.spawn_actor(
            sem_camera_bp,
            carla.Transform(carla.Location(x=2.5, z=1.0)),
            attach_to=self.vehicle
        )
        self.sem_camera.listen(lambda image: self._process_sem_image(image))

        # LiDAR
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        self.lidar = self.world.spawn_actor(
            lidar_bp,
            carla.Transform(carla.Location(x=0, z=2.0)),
            attach_to=self.vehicle
        )
        self.lidar.listen(lambda data: self.process_lidar(data))

        # Датчик столкновений
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self.on_collision(event))

        self.collision_flag = False
        self.episode_reward = 0
        self.last_location = self.vehicle.get_location()
        self.distance = 0
        self.rgb_image = None
        self.sem_image = None
        self.step_counter = 0
        self.obstacle_side = None
        self.pole_detected = False

        # Инициализация окна для семантической камеры
        cv2.namedWindow('Semantic Camera', cv2.WINDOW_AUTOSIZE)

        # Ожидание инициализации изображений
        timeout = time.time() + 5
        while (self.rgb_image is None or self.sem_image is None) and time.time() < timeout:
            self.world.tick()
            time.sleep(0.1)

        if self.rgb_image is None or self.sem_image is None:
            print("[RESET] Не удалось инициализировать изображения, повторяем reset")
            return self.reset()

        self._update_state()
        if self.state is None:
            print("[RESET] Не удалось инициализировать состояние, повторяем reset")
            return self.reset()

        return self.state

    def _process_rgb_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        self.rgb_image = cv2.resize(array, (64, 64)).astype(np.uint8)
        self._update_state()

    def _process_sem_image(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        self.sem_image = cv2.resize(array, (64, 64)).astype(np.uint8)
        display_image = cv2.resize(array, (640, 480))
        cv2.imshow('Semantic Camera', display_image)
        cv2.waitKey(1)
        self._update_state()

    def _update_state(self):
        if self.rgb_image is not None and self.sem_image is not None:
            self.state = np.concatenate([self.rgb_image, self.sem_image], axis=-1)
            writer.add_image('RGB Image', self.rgb_image.transpose(2, 0, 1), global_step=self.step_counter)
            writer.add_image('Semantic Image', self.sem_image.transpose(2, 0, 1), global_step=self.step_counter)
        else:
            self.state = None

    def process_lidar(self, data):
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        distances = np.linalg.norm(points[:, :3], axis=1)
        # Фильтр для препятствий на расстоянии 20–30 см
        close_obstacles = (distances >= 0.2) & (distances <= 0.3) & (points[:, 0] > 0) & (points[:, 2] > 0.5)
        if np.any(close_obstacles):
            self.episode_reward -= 100
            print("[LiDAR] Препятствие на расстоянии 20–30 см! Штраф -100.")

        self.obstacle_side = None
        self.pole_detected = False
        if np.any(close_obstacles):
            obstacle_points = points[close_obstacles]
            mean_y = np.mean(obstacle_points[:, 1])
            if mean_y > 0.1:
                self.obstacle_side = 'right'
                print("[LiDAR] Препятствие справа, планируем объезд налево.")
            elif mean_y < -0.1:
                self.obstacle_side = 'left'
                print("[LiDAR] Препятствие слева, планируем объезд направо.")
            # Проверка на столбы
            if np.any(obstacle_points[:, 2] > 1.0):
                self.pole_detected = True
                print("[LiDAR] Обнаружен столб на расстоянии 20–30 см!")

    def analyze_semantic_image(self):
        if self.sem_image is None:
            return 0, 0, 0

        height, width = self.sem_image.shape[:2]
        left_zone = self.sem_image[:, :width//3]
        center_zone = self.sem_image[:, width//3:2*width//3]
        right_zone = self.sem_image[:, 2*width//3:]

        zones = {'left': left_zone, 'center': center_zone, 'right': right_zone}
        analysis = {zone: {key: 0 for key in self.semantic_colors} for zone in zones}

        for zone_name, zone_data in zones.items():
            for class_name, color in self.semantic_colors.items():
                mask = np.all(zone_data == color, axis=-1)
                analysis[zone_name][class_name] = np.sum(mask) / (zone_data.shape[0] * zone_data.shape[1])

        steer_adjustment = 0
        reward_adjustment = 0
        lane_center_deviation = 0

        # Проверка разметки
        if analysis['center']['lane_marking'] > 0.05:
            reward_adjustment += 30
            print("[SEMANTIC] Разметка в центре, следуем по полосе. Бонус +30.")
        elif analysis['left']['lane_marking'] > analysis['right']['lane_marking'] and analysis['left']['lane_marking'] > 0.02:
            steer_adjustment += 0.5
            lane_center_deviation = 0.2
            print("[SEMANTIC] Разметка слева, поворот направо.")
        elif analysis['right']['lane_marking'] > analysis['left']['lane_marking'] and analysis['right']['lane_marking'] > 0.02:
            steer_adjustment -= 0.5
            lane_center_deviation = 0.2
            print("[SEMANTIC] Разметка справа, поворот налево.")

        # Проверка дороги
        if analysis['center']['road'] > 0.7:
            reward_adjustment += 40
            print("[SEMANTIC] Дорога в центре, едем правильно. Бонус +40.")
        else:
            reward_adjustment -= 60
            lane_center_deviation += 0.3
            print("[SEMANTIC] Дорога не в центре! Штраф -60.")

        # Проверка тротуара
        if analysis['center']['sidewalk'] > 0.1:
            reward_adjustment -= 120
            print("[SEMANTIC] Тротуар в центре! Штраф -120.")
            if analysis['left']['road'] > analysis['right']['road']:
                steer_adjustment += 0.7
                print("[STEER] Поворот направо, чтобы вернуться на дорогу.")
            else:
                steer_adjustment -= 0.7
                print("[STEER] Поворот налево, чтобы вернуться на дорогу.")

        # Проверка столбов и других препятствий
        if analysis['center']['pole'] > 0.05 or analysis['center']['car'] > 0.05 or analysis['center']['pedestrian'] > 0.05:
            reward_adjustment -= 100
            self.pole_detected = True
            print("[SEMANTIC] Препятствие (столб/машина/пешеход) в центре! Штраф -100.")
            if analysis['left']['road'] > analysis['right']['road']:
                steer_adjustment += 0.6
                print("[STEER] Поворот направо для объезда препятствия.")
            else:
                steer_adjustment -= 0.6
                print("[STEER] Поворот налево для объезда препятствия.")

        # Проверка стен и зданий
        if analysis['center']['wall'] > 0.1 or analysis['center']['building'] > 0.1:
            reward_adjustment -= 150
            print("[SEMANTIC] Стена или здание в центре! Штраф -150.")
            if analysis['left']['road'] > analysis['right']['road']:
                steer_adjustment += 0.7
                print("[STEER] Поворот направо, чтобы избежать стены/здания.")
            else:
                steer_adjustment -= 0.7
                print("[STEER] Поворот налево, чтобы избежать стены/здания.")

        return steer_adjustment, reward_adjustment, lane_center_deviation

    def get_next_waypoint(self):
        vehicle_location = self.vehicle.get_transform().location
        waypoints = self.map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        return waypoints.next(2.0)[0]

    def on_collision(self, event):
        self.collision_flag = True
        self.episode_reward -= 200
        print("[COLLISION] Столкновение! Штраф -200.")

    def step(self, action):
        self.step_counter += 1

        # Константы для функции награды
        COLLISION_PENALTY = -200
        DISTANCE_DEVIATION_PENALTY_FACTOR = -20
        ANGLE_DEVIATION_PENALTY_FACTOR = -10
        DISTANCE_BONUS_FACTOR = 0.03
        SPEED_BONUS_FACTOR = 4
        STEERING_PENALTY_FACTOR = 0.4
        SIDEWALK_PENALTY = -120
        OBSTACLE_PENALTY = -100
        OBSTACLE_AVOIDANCE_BONUS = 50
        SPEED_PENALTY = -20
        LANE_MARKING_BONUS = 30

        throttle = np.clip(action[0], 0.2, 1.0)
        steer = np.clip(action[1], -1.0, 1.0)

        # Получение следующей точки маршрута
        waypoint = self.get_next_waypoint()
        target_vector = np.array([
            waypoint.transform.location.x - self.vehicle.get_location().x,
            waypoint.transform.location.y - self.vehicle.get_location().y
        ])
        target_angle = np.arctan2(target_vector[1], target_vector[0])
        vehicle_angle = np.deg2rad(self.vehicle.get_transform().rotation.yaw)
        angle_diff = abs(target_angle - vehicle_angle)

        # Корректировка рулевого управления
        steer_correction = np.clip((target_angle - vehicle_angle) * 2.0, -0.6, 0.6)

        # Обработка данных LiDAR для объезда препятствий
        if self.obstacle_side is not None:
            if self.obstacle_side == 'right':
                steer_correction -= 0.6
                print("[STEER] Поворот налево для объезда препятствия справа (LiDAR).")
            elif self.obstacle_side == 'left':
                steer_correction += 0.6
                print("[STEER] Поворот направо для объезда препятствия слева (LiDAR).")

        # Обработка данных семантической камеры
        sem_steer_adjustment, sem_reward_adjustment, lane_center_deviation = self.analyze_semantic_image()
        steer_correction += sem_steer_adjustment

        # Окончательное значение рулевого управления
        steer = np.clip(steer + steer_correction, -1.0, 1.0)

        # Применение управления
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer, brake=0, gear=1)
        )

        # Расчет скорости
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        # Расчет награды
        reward = 0
        done = False

        if self.collision_flag:
            reward += COLLISION_PENALTY
            done = True

        # Расчет пройденного расстояния
        current_location = self.vehicle.get_location()
        dx = current_location.x - self.last_location.x
        dy = current_location.y - self.last_location.y
        self.distance += np.sqrt(dx**2 + dy**2)
        self.last_location = current_location
        distance_deviation = np.linalg.norm(target_vector) * 0.1

        # Штрафы и бонусы
        reward += DISTANCE_DEVIATION_PENALTY_FACTOR * distance_deviation
        reward += ANGLE_DEVIATION_PENALTY_FACTOR * angle_diff
        reward -= 20 * lane_center_deviation
        reward += DISTANCE_BONUS_FACTOR * self.distance

        # Награда за скорость
        if 5.0 <= speed <= 15.0:
            reward += SPEED_BONUS_FACTOR * speed
        else:
            reward += SPEED_PENALTY
            print("[SPEED] Скорость вне диапазона (5–15 м/с)! Штраф -20.")

        # Награда за объезд препятствий
        if self.obstacle_side is not None:
            points = np.frombuffer(self.lidar.get().raw_data, dtype=np.float32).reshape(-1, 4)
            distances = np.linalg.norm(points[:, :3], axis=1)
            close_obstacles = (distances >= 0.2) & (distances <= 0.3) & (points[:, 0] > 0)
            if not np.any(close_obstacles):
                reward += OBSTACLE_AVOIDANCE_BONUS
                print("[REWARD] Успешный объезд препятствия (LiDAR)! Бонус +50.")
                self.obstacle_side = None
            else:
                reward += OBSTACLE_PENALTY
                print("[REWARD] Препятствие не объезжено! Штраф -100.")

        # Штраф за столбы
        if self.pole_detected:
            reward += OBSTACLE_PENALTY
            print("[REWARD] Столб в зоне, штраф -100.")

        # Награда от семантической камеры
        reward += sem_reward_adjustment

        # Проверка тротуара
        if self.sem_image is not None:
            sidewalk_pixels = np.all(self.sem_image == self.semantic_colors['sidewalk'], axis=-1)
            sidewalk_ratio = np.sum(sidewalk_pixels) / (64 * 64)
            if sidewalk_ratio > 0.1:
                reward += SIDEWALK_PENALTY
                print("[SEMANTIC] На тротуаре! Штраф -120.")

        # Штраф за резкое руление
        reward -= STEERING_PENALTY_FACTOR * abs(steer)

        # Обновление общей награды
        self.episode_reward += reward

        # Логирование в TensorBoard
        writer.add_scalar('Reward', reward, global_step=self.step_counter)
        writer.add_scalar('Total Reward', self.episode_reward, global_step=self.step_counter)
        writer.add_scalar('Speed', speed, global_step=self.step_counter)
        writer.add_scalar('Distance', self.distance, global_step=self.step_counter)
        writer.add_scalar('Lane Deviation', lane_center_deviation, global_step=self.step_counter)

        # Вывод отладочной информации
        print(f"🚗 Скорость: {speed:.2f} м/с | Угол ошибки: {angle_diff:.2f}")
        print(f"🚗 Отклонение от траектории: {distance_deviation:.2f} | Общее расстояние: {self.distance:.2f}")
        print(f"🎯 Обнаружено препятствие? {'Да' if self.obstacle_side is not None else 'Нет'}")
        print(f"🎯 Обнаружен столб? {'Да' if self.pole_detected else 'Нет'}")
        print(f"📊 Текущая награда: {reward:.2f} | Общая награда: {self.episode_reward:.2f}")

        return self.state, reward, done, {}

    def close(self):
        self.destroy_actors()
        cv2.destroyAllWindows()

    def destroy_actors(self):
        if self.rgb_camera:
            self.rgb_camera.stop()
            self.rgb_camera.destroy()
        if self.sem_camera:
            self.sem_camera.stop()
            self.sem_camera.destroy()
        if self.lidar:
            self.lidar.stop()
            self.lidar.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()

class Actor(nn.Module):
    def __init__(self, state_shape, action_dim, max_action):
        super(Actor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        x = state.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.max_action * self.fc(x)

class Critic(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(Critic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4 + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, state, action):
        x = state.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        return self.fc(x)

class DDPG:
    def __init__(self, state_shape, action_dim, max_action):
        self.actor = Actor(state_shape, action_dim, max_action)
        self.actor_target = Actor(state_shape, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)

        self.critic = Critic(state_shape, action_dim)
        self.critic_target = Critic(state_shape, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(1000000)
        self.noise = OUNoise(action_dim)
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        return np.clip(action + self.noise.noise(), [0.2, -self.max_action], [self.max_action, self.max_action])

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        target_q = self.critic_target(next_states, self.actor_target(next_states))
        target_q = rewards + (1 - dones) * self.gamma * target_q
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        writer.add_scalar('Critic Loss', critic_loss.item(), global_step=len(self.replay_buffer))
        writer.add_scalar('Actor Loss', actor_loss.item(), global_step=len(self.replay_buffer))

def main():
    env = CarlaEnvDDPG()
    state_shape = (64, 64, 6)
    action_dim = 2
    max_action = 1.0

    agent = DDPG(state_shape, action_dim, max_action)
    episodes = 100
    max_steps = 1000

    for episode in range(episodes):
        state = env.reset() / 255.0
        print(f"[DEBUG] State shape after reset: {state.shape}")
        agent.noise.reset()
        episode_reward = 0
        step = 0

        while step < max_steps:
            action = agent.select_action(state)
            print(f"[DEBUG] Action: {action}")
            next_state, reward, done, _ = env.step(action)
            next_state = next_state / 255.0
            print(f"[DEBUG] Next state shape: {next_state.shape}")

            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            episode_reward += reward
            step += 1

            if done:
                break

        print(f"Эпизод {episode + 1}/{episodes} завершён! Итоговая награда: {episode_reward:.2f}")
        writer.add_scalar('Episode Reward', episode_reward, global_step=episode)

    writer.flush()
    env.close()

if __name__ == "__main__":
    main()