import numpy as np
import random
import os

foldername = "/Users/tarek/Documents/UNI/Lab Rotations/Kremkow/Data/Stimuli/"
filepath = os.path.join(foldername, 'checkerboard_200.npy') 

# 1. Set Random Seed
np.random.seed(1675)

# 2. Define dimensions
xn = 40
yn = 40

# 3. Create main checker matrix
checker = np.random.randint(2, size=(xn,yn)).astype(float)
checker *= 2.
checker -= 1.

# 4. Flip the checker matrix
checker = np.flip(checker, 0)

# 5. Crop middle 36x22 part.
checker_cropped_x = checker[2:-2]
checker_cropped_xy = checker_cropped_x.T[9:-9]
checker = checker_cropped_xy.T

checker_inverse = checker*-1.

both_checkers = np.array((checker, checker_inverse))
total_checkers = np.tile(both_checkers, (100, 1, 1))

with open(filepath, 'wb') as f:
    np.save(f, total_checkers)


###### The next section uses psychopy to show the images #####
# %%
#n_frames = stimulus_frames.shape[2]
# stim_size = np.array([xn,yn]).astype('float')
# #images = {}
# image_checker = visual.ImageStim(win,checker,size=stim_size*scale,pos=position)
# image_checker_inverse = visual.ImageStim(win,checker_inverse,size=stim_size*scale,pos=position)
# # %%
# # we save the params
# # if not params['test']:
# #     # we save the params in the folder, always needed. Just in test maybe not
# #     np.save(filename_save,params)
# #     # we wait for the online analysis
# #     if params['closed_loop_with_analysis']:
# #         OurSetup.write_current_params_and_wait_for_go(params)

# # we present some text outside the screen to get the system up and running
# # OurSetup.present_pause(120,win,trigger)# in frames
# # TODO: 200
# for trial in range(200):
#     # on even trials we present checker, on odds checker inverse
#     if np.mod(trial,2) == 0: # even
#         image = image_checker
#     else:
#         image = image_checker_inverse
#     # present images
#     for flipN in range(stimulus_duration_in_frames):
#         image.draw()
#         win.flip()
#         #raw_input()
#         if flipN == 0:
#             # trigger.FrameTime()
# #            trigger.EyeCamera()
#             a = 1
# # done
# # we flip one more time to make the screen gray
# win.flip()
# # OurSetup.present_pause(360,win,trigger)# in frames
