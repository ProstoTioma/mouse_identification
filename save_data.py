from pynput import mouse
import keyboard
from scipy.spatial.distance import cdist
import numpy as np


class MouseListener:
    def __init__(self, file_path, file_path_filtered):
        self.file_path = file_path
        self.file_path_filtered = file_path_filtered
        self.mouse_listener = mouse.Listener(on_move=self.on_move)
        self.mouse_positions = []
        self.mouse_listener.start()

    def on_move(self, x, y):
        self.mouse_positions.append((x, y))
        if keyboard.is_pressed('ctrl'):
            self.mouse_listener.stop()
            self.save_to_file(density_threshold)

    def stop(self):
        self.mouse_listener.stop()
        self.mouse_listener.join()

    def save_to_file(self, density_threshold):
        # Remove consecutive duplicate points and filter by density
        filtered_positions = []
        for pos in self.mouse_positions:
            if self.check_density(pos, self.mouse_positions, density_threshold):
                filtered_positions.append(pos)

        with open(self.file_path_filtered, 'w') as file:
            for position in filtered_positions:
                file.write(f"{position[0]}, {position[1]}\n")

        with open(self.file_path, 'w') as file:
            for pos in self.mouse_positions:
                file.write(f"{pos[0]}, {pos[1]}\n")

    def check_density(self, point, existing_points, density_threshold):
        existing_points_array = np.array(existing_points) if existing_points else np.empty((0, 2))
        distances = cdist([point], existing_points_array)
        num_neighbors = np.sum(distances < density_threshold)
        print(num_neighbors)
        return num_neighbors > 20  # Adjust the threshold as needed


if __name__ == "__main__":
    is_me = True
    if is_me:
        file_path_filtered = "mouse_movement_filtered.txt"
        file_path = "mouse_movement.txt"
    else:
        file_path_filtered = "mouse_movement_filtered_fake.txt"
        file_path = "mouse_movement_fake.txt"

    listener = MouseListener(file_path, file_path_filtered)
    density_threshold = 5

    try:
        print("Mouse movement tracking started. Press 'ctrl' to stop.")
        listener.mouse_listener.join()
    except KeyboardInterrupt:
        print("\nMouse movement tracking stopped.")
        listener.stop()
        listener.save_to_file(density_threshold)
