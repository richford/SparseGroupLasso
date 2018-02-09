# Do all the profiling. First line-by-line percentages, then timing:
# Run as `ipython script.py`
get_ipython().magic('load_ext line_profiler')
import profile_sgl as p
from SparseGroupLasso import SGL
get_ipython().magic('lprun -f SGL.fit -f SGL._grad_l -f SGL.discard_group p.func()')
get_ipython().magic('timeit p.func()')
