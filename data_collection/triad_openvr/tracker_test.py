import triad_openvr
import time
import sys
import csv

v = triad_openvr.triad_openvr()
v.print_discovered_objects()

if len(sys.argv) == 1:
    interval = 1/250
elif len(sys.argv) == 2:
    interval = 1/float(sys.argv[1])
else:
    print("Invalid number of arguments")
    interval = False
    


if interval:
    # open CSV file for logging
    with open("tracker_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_sec","x","y","z","yaw","pitch","roll"])
        t0 = time.time()

        while True:
            start = time.time()
            pose = v.devices["tracker_1"].get_pose_euler()  # returns tuple/list
            if pose is not None:
                # print to screen
                txt = " ".join(["%.4f" % each for each in pose])
                print("\r" + txt, end="")

                # write to CSV
                writer.writerow([time.time()-t0] + list(pose))
                f.flush()

            sleep_time = interval-(time.time()-start)
            if sleep_time > 0:
                time.sleep(sleep_time)