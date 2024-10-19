from typing import List
import numpy as np
from scipy.optimize import fsolve


class Pump:
    """Pump component with variable speed."""

    def __init__(self, coeff: List[float]):
        """Initializes a pump component with a flow-head curve.

        Parameters
        ----------
            coeff: List[float]
                Coefficients of the flow-head curve.

        Raises
        ------
            ValueError
                If coeff is not of length 2.
        """
        if len(coeff) != 2:
            raise ValueError("Coefficients must be of length 2.")
        self.coeff = coeff

    def calculate_head(self, flowrate, pump_speed=1.0):
        """Calculates head rise across a pump for a given flow-head curve and pump speed.

        Parameters
        ----------
            flowrate: float
                Flowrate of the pump.
            pump_speed: float, optional
                Speed of the pump.

        Returns
        -------
            float
                Head of the pump.

        Raises
        ------
            ValueError
                If pump_speed is not between 0 and 1.
        """
        if pump_speed < 0 or pump_speed > 1:
            raise ValueError("Speed must be between 0 and 1.")
        head = self.coeff[0] * pump_speed**2 - self.coeff[1] * flowrate**2
        return head if head > 0 else 0


class ControlValve:
    """Control valve with variable opening and valve characterstic."""

    def __init__(self, Kv_max, a_valve=2, characteristic="linear"):
        """Initializes a control valve component with a maximum Kv value and a valve characteristic.

        Parameters
        ----------
            Kv_max: float
                Maximum flow coefficient of the valve.
            a_valve: float, optional
                Exponent of the valve characteristic curve.
            characteristic: str, optional
                Characteristic curve of the valve.

        Raises
        ------
            ValueError
                If characteristic is not one of self.CHARACTERSTIC_CURVES keys.
        """
        self.a_valve = a_valve
        self.Kv_max = Kv_max
        self.CHARACTERISTIC_CURVES = {
            "linear": lambda pos, maximum: pos * maximum,
            "equal_percentage": lambda pos, maximum: pos**a_valve * maximum,
            "quick_opening": lambda pos, maximum: pos ** (1 / a_valve) * maximum,
        }
        if characteristic not in self.CHARACTERISTIC_CURVES:
            raise ValueError(
                f"Invalid characteristic. Must be one of {self.CHARACTERISTIC_CURVES.keys()}"
            )
        self.characteristic = characteristic

    def calculate_Kv(self, valve_position):
        """Calculates the flow coefficient of the valve for a given valve position.

        Parameters
        ----------
            valve_position: float
                Position of the valve (0 - 1).

        Returns
        -------
            float
                Flow coefficient of the valve in m^3/h.

        Raises
        ------
            ValueError
                If valve_position is not between 0 and 1.
        """
        if valve_position < 0 or valve_position > 1:
            raise ValueError("Valve position must be between 0 and 1.")
        if valve_position == 0:
            return 0  # Valve is fully closed
        return self.CHARACTERISTIC_CURVES[self.characteristic](
            valve_position, self.Kv_max
        )

    def calculate_pressure_drop(self, flowrate, valve_position, density=1000, g=9.81):
        """Calculates the pressure drop across the valve for a given flowrate and valve position.

        Parameters
        ----------
            flowrate: float
                Flowrate of the fluid in m3/s.
            valve_position: float
                Position of the valve (0 - 1).
            density: float, optional
                Density of the fluid in kg/m3. Default is 1000.
            g: float, optional
                Acceleration due to gravity in m/s^2. Default is 9.81.

        Returns
        -------
            float
                Pressure drop across the valve in Pa.
        """
        Kv = self.calculate_Kv(valve_position)
        specific_gravity = density / 1000  # for liquids
        if Kv < 1e-3:
            return 1e12  # very high pressure drop for closed valve
        return (flowrate * 3600 / Kv) ** 2 * (specific_gravity) * 10**5

    def calculate_head_loss(self, flowrate, valve_position, density=1000, g=9.81):
        """Calculates the head loss across the valve for a given flowrate and valve position.

        Parameters
        ----------
            flowrate: float
                Flowrate of the fluid in m3/s.
            valve_position: float
                Position of the valve (0 - 1).
            density: float, optional
                Density of the fluid in kg/m3. Default is 1000.
            g: float, optional
                Acceleration due to gravity in m/s^2. Default is 9.81.

        Returns
        -------
            float
                Head loss across the valve in m.
        """
        return self.calculate_pressure_drop(flowrate, valve_position, density, g) / (
            density * g
        )


class Pipe:
    """Simplified pipework component with constant resistance."""

    def __init__(self, K_pipe):
        """Initializes a pipe component with a constant resistance.

        Parameters
        ----------
            K_pipe: float
                Resistance of the pipe in Pa.s^2/m^3.
        """
        self.K_pipe = K_pipe

    def calculate_resistance(self):
        """Calculates the resistance of the pipe.

        Returns
        -------
            float
                Resistance of the pipe in Pa.s^2/m^3.
        """
        return self.K_pipe

    def calculate_head_loss(self, flowrate):
        """Calculates the head loss across the pipe for a given flowrate.

        Parameters
        ----------
            flowrate: float
                Flowrate of the fluid in m3/s.

        Returns
        -------
            float
                Head loss across the pipe in m.
        """
        return self.K_pipe * flowrate**2


class SimplePipingSystem:
    def __init__(self, pump: Pump, valve: ControlValve, pipe: Pipe):
        self.pump = pump
        self.valve = valve
        self.pipe = pipe

    def calculate_flowrate(self, pump_speed: float, valve_position: float) -> float:
        """Calculates flowrate of a simple piping system given pump and system head.

        Parameters
        ----------
            pump_speed: float
                Speed of the pump.
            valve_position: float
                Position of the control valve.

        Returns
        -------
            float
                Flowrate of the piping system in m3/s."""
        if pump_speed < 1e-3 or valve_position < 1e-3:
            # no flow if pump speed or valve position is almost zero
            return 0.0

        def objective(flowrate):
            head_pump = self.pump.calculate_head(flowrate, pump_speed)
            head_valve = self.valve.calculate_head_loss(flowrate, valve_position)
            head_pipe = self.pipe.calculate_head_loss(flowrate)
            head_system = head_valve + head_pipe
            return (
                head_pump - head_system
            )  # system head will be equal to pump head in ideal conditions

        flowrate_solution, _, ier, _ = fsolve(
            objective, 1.0, full_output=True, maxfev=1000, xtol=1e-5
        )
        if ier != 1 or flowrate_solution[0] < 0:
            return 0.0
        return flowrate_solution[0]
