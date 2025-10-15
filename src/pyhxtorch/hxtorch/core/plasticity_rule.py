from __future__ import annotations

from math import ceil

import pygrenade_vx.network.abstract as gabstract
from dlens_vx_v3 import hal


class Timer:
    """
    Periodic timer information for plasticity rule execution.
    EA: Copied from PyNN
    """
    def __init__(
        self,
        **parameters,
    ):
        self._start = parameters["start"]
        self._period = parameters["period"]
        self._num_periods = parameters["num_periods"]
        self.parameters = {
            x: param for x, param in parameters.items()
            if x not in ["start", "period", "num_periods"]
        }

    def _set_start(self, new_start):
        self._start = new_start

    def _get_start(self):
        return self._start

    def _set_period(self, new_period):
        self._period = new_period

    def _get_period(self):
        return self._period

    def _set_num_periods(self, new_num_periods):
        self._num_periods = new_num_periods

    def _get_num_periods(self):
        return self._num_periods

    start = property(_get_start, _set_start)
    period = property(_get_period, _set_period)
    num_periods = property(_get_num_periods, _set_num_periods)

    def to_grenade(
        self,
        snippet_begin_time: float,
        snippet_end_time: float,
    ) -> gabstract.PlasticityRule.Dynamics.Timer:
        def to_ppu_cycles(value: float) -> int:
            result = float(value)
            result = result * float(hal.Timer.Value.fpga_clock_cycles_per_us)
            result = result * 2  # 250MHz vs. 125MHz
            return gabstract.PlasticityRule.Dynamics.Timer.Value(
                int(round(result))
            )

        # Snippet boundaries are provided in seconds, while timer
        # configuration uses microseconds.
        snippet_begin_time_us = snippet_begin_time * 1e6
        snippet_end_time_us = snippet_end_time * 1e6

        timer = gabstract.PlasticityRule.Dynamics.Timer()
        pre_snippet_period_count = ceil(
            max(snippet_begin_time_us - self.start, 0) / self.period
        )
        timer.start = to_ppu_cycles(
            self.period * pre_snippet_period_count + self.start
        )
        timer.period = to_ppu_cycles(self.period)
        timer.num_periods = min(
            self.num_periods,
            ceil(
                max(snippet_end_time_us - self.start, 0) / self.period
            ) - pre_snippet_period_count,
        )

        return timer


class PlasticityRule:

    def __init__(
        self,
        kernel: str,
        timer: Timer,
    ):
        self.timer = timer
        self._kernel = kernel

    def generate_kernel(self):
        return self._kernel
