from typing import List
from lib.mputils import consume_queue, produce_queue, write_list_to_queue
import unittest
import multiprocessing as mp
from hypothesis import given, settings, strategies as st


class TestMPUtils(unittest.TestCase):
    @given(st.lists(st.integers()))
    @settings(max_examples=20)
    def test_produce_consume_queues(self, l: List[int]):
        q = produce_queue(l)
        ll = consume_queue(q)
        self.assertEqual(l, ll)

    @given(st.lists(st.integers()))
    @settings(max_examples=20)
    def test_multiproc_queues(self, l: List[int]):
        q: "mp.Queue[int]" = mp.Queue()
        producer_proc = mp.Process(
            target=write_list_to_queue,
            args=(
                l,
                q,
            ),
        )
        producer_proc.start()
        ll = consume_queue(q)
        producer_proc.join()
        self.assertEqual(l, ll)
