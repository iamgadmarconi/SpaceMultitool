import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D


def plot_3d(model, time_step, total_time):
    space = model
    space.simulate(total_time, time_step)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Function to update the plot
    def update_plot(val):
        ax.clear()
        t = int(val)  # Use the value of the slider
        for body in space.bodies:
            pos = body.get_position_at_time(t)
            ax.scatter(pos[0], pos[1], pos[2], label=body.name)

        ax.legend()
        plt.draw()

    # Slider
    ax_slider = plt.axes([0.1, 0.02, 0.8, 0.03])
    slider = Slider(ax_slider, 'Time', 0, total_time, valinit=0, valstep=time_step)

    # Update the plot when the slider is changed
    slider.on_changed(update_plot)  # Call the update_plot function

    # Initial plot
    update_plot(0)  # Call the update_plot function with initial value

    plt.show()