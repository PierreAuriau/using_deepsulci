from using_deepsulci.processes.label_resample import LabelResample
from soma import aims


skeleton = "/home/bastien/data/archi/t1-1mm-1/025/t1mri/default_acquisition/" \
           "default_analysis/segmentation/Lskeleton_025.nii.gz"
resampled_skeleton = "/var/tmp/resampled_Lskeleton_025.nii.gz"

vol = aims.read(skeleton)
print(vol.header())

proc = LabelResample()
proc.input_image = skeleton
proc.sx = 2
proc.sy = 2
proc.sz = 2
proc.output_image = resampled_skeleton

proc._run_process()

rvol = aims.read(resampled_skeleton)
print(rvol.header())
