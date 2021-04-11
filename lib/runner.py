from lib.components.follower import Follower
from lib.components.backend import Backend
from lib.mputils import consume_queue_into_conn
from lib.components.audiopreprocessor import AudioPreprocessor
from lib.components.synthesiser import Synthesiser
from lib.args import Arguments
from lib.midi import process_midi_to_note_info
from lib.sharedtypes import (
    ExtractedFeature,
    ExtractedFeatureQueue,
    FollowerOutputQueue,
    MultiprocessingConnection,
    NoteInfo,
)
from typing import Tuple, List
from lib.eprint import eprint
import multiprocessing as mp
import time
import numpy as np


class Runner:
    def __init__(self, args: Arguments):
        """
        Precondition: assuming args.sanitize() was called.
        """
        self.args = args
        self.__log(f"Initiated with arguments:\n{args}")

    def start(self):
        self.__log(f"STARTING")

        follower_output_queue: FollowerOutputQueue = mp.Queue()
        (
            parent_performance_stream_start_conn,
            child_performance_stream_start_conn,
        ) = mp.Pipe()
        P_queue: ExtractedFeatureQueue = mp.Queue()

        self.__log(f"Begin: preprocess score")
        score_note_onsets, S = self.__preprocess_score()
        self.__log(f"End: preprocess score")

        self.__log(f"Begin: initialise performance processor")
        perf_ap = self.__init_performance_processor(P_queue)
        self.__log(f"End: initialise performance processor")

        self.__log(f"Begin: initialise follower")
        follower = self.__init_follower(follower_output_queue, P_queue, S)
        self.__log(f"End: initialise follower")

        self.__log(f"Begin: initialise backend")
        backend = self.__init_backend(
            follower_output_queue,
            parent_performance_stream_start_conn,
            score_note_onsets,
        )
        self.__log(f"End: initialise backend")

        perf_ap_proc = mp.Process(target=perf_ap.start)
        follower_proc = mp.Process(target=follower.start)
        backend_proc = mp.Process(target=backend.start)

        # start from the back
        self.__log(f"Starting: backend")
        backend_proc.start()
        self.__log(f"Starting: follower")
        follower_proc.start()

        perf_start_time = time.perf_counter()
        self.__log(f"Starting: performance at {perf_start_time}")
        child_performance_stream_start_conn.send(perf_start_time)
        perf_ap_proc.start()

        perf_ap_proc.join()
        self.__log("Joined: performance")
        follower_proc.join()
        self.__log("Joined: follower")
        backend_proc.join()
        self.__log("Joined: backend")

    def __init_performance_processor(
        self, P_queue: ExtractedFeatureQueue
    ) -> AudioPreprocessor:
        args = self.args

        ap = AudioPreprocessor(
            args.sample_rate,
            args.perf_wave_path,
            args.slice_len,
            self.__get_frame_length(),
            args.simulate_performance,
            args.mode,
            args.cqt,
            args.fmin,
            args.fmax,
            args.slice_len,
            args.transition_slice_ratio,
            P_queue,
        )

        return ap

    def __init_backend(
        self,
        follower_output_queue: FollowerOutputQueue,
        performance_stream_start_conn: MultiprocessingConnection,
        score_note_onsets: List[NoteInfo],
    ) -> Backend:
        args = self.args

        def output_func(s: str):
            print(s, flush=True)

        return Backend(
            args.backend,
            follower_output_queue,
            performance_stream_start_conn,
            score_note_onsets,
            args.slice_len,
            args.sample_rate,
            output_func,
        )

    def __init_follower(
        self,
        follower_output_queue: FollowerOutputQueue,
        P_queue: ExtractedFeatureQueue,
        S: List[ExtractedFeature],
    ) -> Follower:
        args = self.args
        return Follower(
            args.mode,
            args.dtw,
            args.max_run_count,
            args.search_window,
            follower_output_queue,
            P_queue,
            S,
        )

    def __preprocess_score(self) -> Tuple[List[NoteInfo], List[ExtractedFeature]]:
        """
        Return note_onsets and features extracted
        """
        args = self.args

        note_onsets = process_midi_to_note_info(args.score_midi_path)
        self.__log("Finished getting note onsets from score midi")

        synthesiser = Synthesiser(args.score_midi_path, args.sample_rate)
        score_wave_path = synthesiser.synthesise()
        self.__log(f"Score midi synthesised to {score_wave_path}")

        S_queue: ExtractedFeatureQueue = ExtractedFeatureQueue(mp.Queue())
        # need to consume into a connection--queues are likely to fill up and reach their
        # limit then cause the program to hang!
        parent_S_conn, child_S_conn = mp.Pipe()
        consume_S_queue_proc = mp.Process(
            target=consume_queue_into_conn, args=(S_queue, child_S_conn)
        )
        consume_S_queue_proc.start()

        audio_preprocessor = AudioPreprocessor(
            args.sample_rate,
            score_wave_path,
            args.slice_len,
            self.__get_frame_length(),
            False,
            args.mode,
            args.cqt,
            args.fmin,
            args.fmax,
            args.slice_len,
            args.transition_slice_ratio,
            S_queue,
        )
        audio_preprocessor.start()

        S = parent_S_conn.recv()

        consume_S_queue_proc.join()
        return (note_onsets, S)

    def __get_frame_length(self) -> int:
        args = self.args
        if args.cqt == "nsgt":
            return args.transition_slice_ratio * args.slice_len
        return args.slice_len

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
