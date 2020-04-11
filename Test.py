import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath('matlab')
eng.thesis_distance_learning(nargout=0)
