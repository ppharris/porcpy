#!/usr/bin/env python

from porcpy.diag_scripts import hot_days as hd


def test_is_in_period():

    jja = [6, 7, 8]

    for false in [-1, 0, 1, 12, 13]:
        assert hd.is_in_period(false, jja) == False

    for true in [6, 7, 8]:
        assert hd.is_in_period(true, jja) == True

    assert hd.is_in_period(range(-1, 14), jja) == [False, False, False,
                                                   False, False, False,
                                                   False, True, True,
                                                   True, False, False,
                                                   False, False, False]

    assert hd.is_in_period(jja, jja) == [True, True, True]
