import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # Use TkAgg backend (replace with a different backend if needed)


def plot_mouse_movement(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    mouse_positions = [tuple(map(int, line.strip().split(','))) for line in lines]

    x_positions, y_positions = zip(*mouse_positions)

    plt.plot(x_positions, y_positions, marker='o', linestyle='-', markersize=1)
    plt.title('Mouse Movement')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.gca().invert_yaxis()
    plt.xlim((0, 2560))
    plt.ylim((0, 1440))

    plt.show()


if __name__ == "__main__":
    is_me = True
    if is_me:
        file_path_filtered = "mouse_movement_filtered.txt"
        file_path = "mouse_movement.txt"
    else:
        file_path_filtered = "mouse_movement_filtered_fake.txt"
        file_path = "mouse_movement_fake.txt"
    plot_mouse_movement(file_path)
    plot_mouse_movement(file_path_filtered)
