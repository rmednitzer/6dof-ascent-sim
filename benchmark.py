import timeit

from sim.dynamics.flex_body import FlexBody
from sim.dynamics.slosh import SloshModel


def benchmark_flex_body():
    fb = FlexBody()

    def test_velocities():
        return fb.modal_velocities()

    def test_displacements():
        return fb.modal_displacements()

    def test_update():
        fb.update(dt=0.01, tvc_force_n=1000.0, propellant_fraction=0.8)

    n = 100000
    t_vel = timeit.timeit(test_velocities, number=n)
    t_disp = timeit.timeit(test_displacements, number=n)
    t_upd = timeit.timeit(test_update, number=n)

    print(f"FlexBody (n={n}):")
    print(f"  modal_velocities:    {t_vel:.4f}s ({t_vel / n * 1e6:.2f} µs/call)")
    print(f"  modal_displacements: {t_disp:.4f}s ({t_disp / n * 1e6:.2f} µs/call)")
    print(f"  update:              {t_upd:.4f}s ({t_upd / n * 1e6:.2f} µs/call)")


def benchmark_slosh():
    sm = SloshModel(n_tanks=2)

    def test_angles():
        return sm.pendulum_angles()

    def test_rates():
        return sm.pendulum_rates()

    def test_update():
        sm.update(dt=0.01, lateral_accel_mps2=5.0, propellant_mass_kg=100000.0, propellant_fraction=0.8)

    n = 100000
    t_ang = timeit.timeit(test_angles, number=n)
    t_rate = timeit.timeit(test_rates, number=n)
    t_upd = timeit.timeit(test_update, number=n)

    print(f"SloshModel (n={n}):")
    print(f"  pendulum_angles: {t_ang:.4f}s ({t_ang / n * 1e6:.2f} µs/call)")
    print(f"  pendulum_rates:  {t_rate:.4f}s ({t_rate / n * 1e6:.2f} µs/call)")
    print(f"  update:          {t_upd:.4f}s ({t_upd / n * 1e6:.2f} µs/call)")


if __name__ == "__main__":
    benchmark_flex_body()
    benchmark_slosh()
