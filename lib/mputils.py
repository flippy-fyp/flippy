from lib.sharedtypes import MultiprocessingConnection
import multiprocessing as mp
from typing import Optional, Any, List, NewType

AnyOptionalQueue = NewType("AnyOptionalQueue", "mp.Queue[Optional[Any]]")


def consume_queue_into_conn(q: AnyOptionalQueue, conn: MultiprocessingConnection):
    """
    Consume queue into a connection
    """
    l = consume_queue(q)
    conn.send(l)


def consume_queue(q: AnyOptionalQueue) -> List[Any]:
    """
    Consume a queue until the end and put the contents into a List
    until the end (a None) is reached.

    The inner type cannot be `Optional`.
    """
    l: List[Any] = []
    while True:
        x = q.get()
        if x is None:
            return l
        l.append(x)


def produce_queue(l: List[Any]) -> AnyOptionalQueue:
    """
    Produce a queue given a list and write it with an ending None.

    The inner type cannot be `Optional`.
    """
    q: AnyOptionalQueue = AnyOptionalQueue(mp.Queue())
    write_list_to_queue(l, q)
    return q


def write_list_to_queue(l: List[Any], q: AnyOptionalQueue):
    """
    Write to a queue from a list and end with an ending None.

    The inner type cannot be `Optional`.
    """
    for x in l:
        q.put(x)
    q.put(None)
