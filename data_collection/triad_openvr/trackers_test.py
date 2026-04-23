import triad_openvr
import time
import sys
import csv

v = triad_openvr.triad_openvr()
v.print_discovered_objects()

# interval
if len(sys.argv) == 1:
    interval = 1/250
elif len(sys.argv) == 2:
    interval = 1/float(sys.argv[1])
else:
    print("Invalid number of arguments")
    sys.exit(1)

# detect trackers
has_t1 = "tracker_1" in v.devices
has_t2 = "tracker_2" in v.devices
if not (has_t1 or has_t2):
    print("No trackers found")
    sys.exit(1)
print(f"tracker_1: {has_t1}, tracker_2: {has_t2}")

# open CSV file for logging
with open("trackers_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    
    # write the header
    header = ["t_sec"]
    if has_t1:
        header += ["t1_x","t1_y","t1_z","t1_yaw","t1_pitch","t1_roll"]
    if has_t2:
        header += ["t2_x","t2_y","t2_z","t2_yaw","t2_pitch","t2_roll"]
    writer.writerow(header)

    t0 = time.time()

    while True:
        start = time.time()

        # write the  row
        row = [time.time() - t0]
        if has_t1:
            pose_1 = v.devices["tracker_1"].get_pose_euler() # returns tuple/list
            row += list(pose_1) if pose_1 else [None]*6
        if has_t2:
            pose_2 = v.devices["tracker_2"].get_pose_euler()
            row += list(pose_2) if pose_2 else [None]*6
        print(row)
        writer.writerow(row)
        f.flush()

        sleep_time = interval-(time.time()-start)
        if sleep_time > 0:
            time.sleep(sleep_time)
