OPENBLAS_NUM_THREADS=1

i = 0

def run(algorithm='BeamformerBase', inputfile_path="../signal_audio/wall1.wav"):

    from pathlib import Path
    import acoular as ac
    import numpy as np

    ac.config.global_caching = 'none'

    sfreq = 96000
    duration = 1
    nsamples = duration * sfreq
    
    micgeofile = Path('../setup/microphones_set.xml')
    inputfile = Path(inputfile_path)
    
    from scipy.io import wavfile
    
    samplerate, wavData = wavfile.read(inputfile)
    ts = ac.TimeSamples(data=wavData, sample_freq=samplerate)
    
    mg = ac.MicGeom(file=micgeofile)
    rg = ac.RectGrid(
        x_min=-2,
        x_max=+2,
        y_min=-2,
        y_max=+2,
        z=1,
        increment=0.1,
    )
    st = ac.SteeringVector(grid=rg, mics=mg)
    
    frg_span = 0.2
    FPS = 30
    frames_count = int(ts.num_samples / ts.sample_freq * FPS)
    frame_length = int(ts.sample_freq / FPS)
    print("Frames to be generated: ", frames_count)
    
    gen = ts.result(frame_length)
    
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots()

    def init():
        ax.clear()
        ax.axis("off")

    def update(frame):
        global i
        i += 1
        print(f"\rFrame {i}/{frames_count}", end="", flush=True)
        res = frame[0]
        p = frame[1]
        fres = frame[2]
        fp = frame[3]
        ax.clear()
        ax.imshow(
            np.transpose(res),
            extent=rg.extend(),
            origin="lower",
        )
        ax.imshow(
            np.transpose(fres),
            extent=(p[0] - frg_span, p[0] + frg_span, p[1] - frg_span, p[1] + frg_span),
            origin="lower",
        )
        ax.plot(fp[0], fp[1], 'r+')
        ax.annotate(f'({fp[0]:0,.2f}, {fp[1]:0,.2f})',
                    xy=(fp[0], fp[1]),
                    xytext=(fp[0] + frg_span, fp[1] + frg_span),
                    color='white')
        
    def mapIndexToRange(i, num, v_min=0, v_max=1):
        step = (v_max - v_min) / (num - 1)
        return v_min + (i * step)

    import time

    t1 = time.thread_time()
    pt = 0
    frames = list()
    total_frame_time = 0
    min_frame_time = 0
    max_frame_time = 0

    for block in gen:
        pt1 = time.thread_time()
        perf_counter_start = time.perf_counter()

        global i
        tempData = block
        tempTS = ac.TimeSamples(data=tempData, sample_freq=samplerate)
        ps = ac.PowerSpectra(source=tempTS, block_size=128, overlap='50%', window='Hanning')

        if algorithm == 'BeamformerBase':
            bb = ac.BeamformerBase(freq_data=ps, steer=st)
        elif algorithm == 'BeamformerFunctional':
            bb = ac.BeamformerFunctional(freq_data=ps, steer=st)
        elif algorithm == 'BeamformerMusic':
            bb = ac.BeamformerMusic(freq_data=ps, steer=st)
        elif algorithm == 'BeamformerCapon':
            bb = ac.BeamformerCapon(freq_data=ps, steer=st)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        tempRes = np.sum(bb.result[4:32], 0)
        r = tempRes.reshape(rg.shape)
        p = np.unravel_index(np.argmax(r), r.shape)
        px = mapIndexToRange(p[0], r.shape[0], rg.extend()[0], rg.extend()[1])
        py = mapIndexToRange(p[1], r.shape[1], rg.extend()[2], rg.extend()[3])
        
        pt2 = time.thread_time()
        pt += pt2 - pt1
        
        frg = ac.RectGrid(
            x_min = px - frg_span,
            x_max = px + frg_span,
            y_min = py - frg_span,
            y_max = py + frg_span,
            z=1,
            increment=0.01,
        )
        fst = ac.SteeringVector(grid=frg, mics=mg, steer_type='classic')

        if algorithm == 'BeamformerBase':
            bf = ac.BeamformerBase(freq_data=ps, steer=fst)
        elif algorithm == 'BeamformerFunctional':
            bf = ac.BeamformerFunctional(freq_data=ps, steer=fst)
        elif algorithm == 'BeamformerMusic':
            bf = ac.BeamformerMusic(freq_data=ps, steer=fst)
        elif algorithm == 'BeamformerCapon':
            bf = ac.BeamformerCapon(freq_data=ps, steer=fst)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        tempFRes = np.sum(bf.result[8:16], 0)
        fr = tempFRes.reshape(frg.shape)
        fp = np.unravel_index(np.argmax(fr), fr.shape)
        fpx = mapIndexToRange(fp[0], fr.shape[0], frg.extend()[0], frg.extend()[1])
        fpy = mapIndexToRange(fp[1], fr.shape[1], frg.extend()[2], frg.extend()[3])

        frames.append((r, (px, py), fr, (fpx, fpy)))
        perf_counter_stop = time.perf_counter() - perf_counter_start
        total_frame_time += perf_counter_stop
        max_frame_time = max(max_frame_time, perf_counter_stop)
        if (i == 0):
            min_frame_time = perf_counter_stop
        min_frame_time = min(min_frame_time, perf_counter_stop)
        print(f"\rBF: {i}", end="", flush=True)
        i += 1

    print()
    
    t2 = time.thread_time()

    avg_frame_time = total_frame_time / i

    print("First stage (low res) time: ", pt, 's')
    print("Second stage (high res) time: ", t2 - t1, 's')
    print("Total frame time: ", total_frame_time, 's')
    print("Average frame time: ", avg_frame_time, 's')
    print("Max frame time: ", max_frame_time, 's')
    print("Min frame time: ", min_frame_time, 's')

    i = 0

    with open(Path("../results/bf_time") / "times.log", "a") as f:
        f.write(f"{inputfile.stem},{algorithm},{pt},{t2 - t1},{total_frame_time},{avg_frame_time},{max_frame_time},{min_frame_time}\n")
    
    points = np.array([ p[1] for p in frames ])
    focus_points = np.array([ p[3] for p in frames ])

    np.save(f"../points_bf/{inputfile.stem}_{algorithm}_points.npy", points)
    np.save(f"../points_bf/{inputfile.stem}_{algorithm}_focuspoints.npy", focus_points)

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, repeat=True, interval=1 / FPS)
    ani.save(f"../results/maps/{inputfile.stem}_{algorithm}_map.mp4", writer="ffmpeg", fps=FPS)
    plt.close()
    i = 0

if __name__ == '__main__':
    algorithms = ['BeamformerBase', 'BeamformerFunctional', 'BeamformerMusic', 'BeamformerCapon']
    recordings = [
        "../signal_audio/wall1.wav",
        "../signal_audio/wall2.wav",
        "../signal_audio/wall3.wav"
    ]

    for rec in recordings:
        for algo in algorithms:
            print(f"\nProcessing file: {rec} with algorithm: {algo}")
            run(algorithm=algo, inputfile_path=rec)