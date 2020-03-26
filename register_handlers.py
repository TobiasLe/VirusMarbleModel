import bpy
import numpy as np
from pathlib import Path
import math

day_count = bpy.data.objects["day_count"]

FRAMES_PER_DAY = np.load(Path(bpy.data.filepath).parent / 'frames_per_day.npy')
def recalculate_text(scene):
    day_count.data.body = 'day: ' + str(math.floor(scene.frame_current / FRAMES_PER_DAY))
bpy.app.handlers.frame_change_pre.append(recalculate_text)

text_n_infected = bpy.data.objects["n_infected"]
infected_for_text = np.load(Path(bpy.data.filepath).parent / 'infected_frame_res.npy')
def recalculate_infected_text(scene):
    text_n_infected.data.body = 'infected: ' + str(int(infected_for_text[scene.frame_current]))
bpy.app.handlers.frame_change_pre.append(recalculate_infected_text)
