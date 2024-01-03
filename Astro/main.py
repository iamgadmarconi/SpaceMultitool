from astromodel_v1 import Space as SM1
from viewer import plot_3d
from bodies import construct_bodies


def main():
    """
    Main function.
    """
    bodies = construct_bodies()
    sm = SM1(bodies)
    time_step = 100000
    total_time = 1000000000

    plot_3d(sm, time_step, total_time)


if __name__ == "__main__":
    main()
