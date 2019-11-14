"""
Module defining a dry spell event class.
"""

from __future__ import print_function, division

from . import FMDI


class Event(object):
    def __init__(self, land, start_index, ut, duration, antep):
        if land < 0:
            raise ValueError("Land point indexes must be non-negative.")
        if duration < 0:
            raise ValueError("Dry spell duration must be at least one day.")
        if antep < 0.0 and antep != FMDI:
            raise ValueError("Antecedent precipitation must not be negative.")

        self.land = land
        self.start_index = start_index
        self.duration = duration
        self.antep = antep
        self._ut = ut

    def __repr__(self):
        return "%d: Start date %s: %d days, %g mm" % (
            self.land, self.start_date(), self.duration, self.antep)

    def start_date(self):
        return self._ut.num2date(self.start_index)

    def end_date(self):
        return self._ut.num2date(self.start_index+self.duration-1)


def iterevents(events):
    """Generator for iterating over all events in groups of full domain
    events lists."""
    for eve_group in events:
        for eve_land in eve_group:
            for eve in eve_land:
                yield eve
