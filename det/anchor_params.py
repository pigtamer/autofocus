import numpy as np
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

retina_size = [list(np.linspace(0.05, 1.05, 5))]
retina_num_anchors = len(retina_size[0]) + len(ratios[0]) - 1