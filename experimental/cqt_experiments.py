from nsgt import CQ_NSGT_sliced, NSGT_sliced, OctScale  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from pysndfile import PySndfile

if __name__ == "__main__":
    print("HELLO")

    sf = PySndfile("./prelude-short.wav")
    fs = sf.samplerate()
    samples = sf.frames()

    s = sf.read_frames(samples)

    if s.ndim > 1:
        s = np.mean(s, axis=1)

    # piano
    scl = OctScale(27.5, 4186, 12)
    slicq = NSGT_sliced(scl, 2 ** 12, 1024, fs)

    # c = []

    # slicelen = 2 ** 11

    # for i in range(0, len(s) + 1, slicelen):
    #     next_c = slicq.forward((s[i : i + slicelen],))
    #     next_c = list(next_c)
    #     c += next_c

    c = slicq.forward((s,))

    # print(c[0])
    print("Plotting t*f space")

    tr = np.array([[np.mean(np.abs(cj)) for cj in ci] for ci in c])
    plt.imshow(
        (np.flipud(tr.T)),
        aspect=float(tr.shape[0]) / tr.shape[1] * 0.5,
        interpolation="nearest",
    )
    plt.savefig("./test3.pdf")