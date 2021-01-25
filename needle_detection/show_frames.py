import numpy as np
import matplotlib.pyplot as plt
import parameters as p ##hier auch später umschreiben als klasse
import time

# Load initialisation image
file = open(p.status_path, 'r')
frame_ptr = file.read()
file.close()
frame_raw = np.load("./saved_frames/"+str(frame_ptr)+".npy")
     
# Prepare plot canvas
fig, ax = plt.subplots()
im = ax.imshow(frame_raw, cmap='gray', animated=True)
plt.show(block=False)
plt.pause(0.5)
bg = fig.canvas.copy_from_bbox(fig.bbox)
ax.draw_artist(im)
fig.canvas.blit(fig.bbox)

t1 = time.time()
try:
    while True:
        
        print(time.time()-t1, "s since last frame")
        t1 = time.time();
        
        # Read current image pointer
        file = open(p.status_path, 'r')
        frame_ptr = file.read()
        file.close()
        
        # Plausability check
        if frame_ptr:
            if int(frame_ptr) >= 0 and int(frame_ptr) <= 15:
                
                # Load image
                frame_raw = np.load("./saved_frames/"+str(frame_ptr)+".npy")
                print(frame_ptr, "     ", frame_raw.shape)
                
                # Update canvas
                fig.canvas.restore_region(bg)
                im.set_data(frame_raw)
                ax.draw_artist(im)
                fig.canvas.blit(fig.bbox)
                fig.canvas.flush_events()
            else:
                print("Wrong ptr")
        else:
            print("Empty file")

except KeyboardInterrupt:
    print("Exiting!")
    pass      